//! The physics engine: a thin, Minecraft-aware layer on top of `rapier3d`.
//!
//! This module has **no dependency on Pumpkin**. The server glue implements
//! [`WorldView`] (block lookups) and calls [`Engine::tick`] once per game tick;
//! the engine hands back [`BodyUpdate`]s describing where every block display
//! should be drawn. That split is what lets the whole thing be unit-tested
//! natively with a fake voxel world.
//!
//! # Model
//!
//! * Every spawned block is a rigid body whose origin is the **centre of the
//!   block** (so rotating the body rotates the visual about the right point).
//!   Non-full blocks (slabs, stairs, fences...) become compound colliders made
//!   from the block's vanilla collision boxes, so their centre of mass, inertia
//!   and shape are physically correct.
//! * World blocks become *static* colliders, created lazily only near bodies
//!   and refreshed while bodies are near, so breaking/placing blocks under a
//!   physics block makes it react.
//! * Friction comes from vanilla block `slipperiness` (ice is slippery, slime is
//!   bouncy), density from the block material (wood floats, iron sinks).
//! * Water and lava apply buoyancy at sampled points inside the submerged
//!   volume (so floating bodies right themselves naturally) plus viscous drag.
//! * Unloaded chunks are treated as solid so nothing falls into the void while
//!   terrain is missing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rapier3d::prelude::*;

use crate::materials::{LAVA_DENSITY, WATER_DENSITY};

/// Integer block coordinates.
pub type Cell = [i32; 3];
/// Stable identifier of a physics body.
pub type BodyId = u64;

/// Liquid kinds that affect bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fluid {
    /// Water (also waterlogged blocks).
    Water,
    /// Lava.
    Lava,
}

/// Everything the engine needs to know about one block state. Built once per
/// distinct state id by the [`WorldView`] and cached.
#[derive(Clone, Debug)]
pub struct StateInfo {
    /// Collision boxes in block-local coordinates `[minx,miny,minz,maxx,maxy,maxz]` (0..1).
    pub boxes: Vec<[f32; 6]>,
    /// Outline boxes; used as the body shape for blocks that have no collision (torches, flowers).
    pub outline: Vec<[f32; 6]>,
    /// Fluid occupying this cell, if any.
    pub fluid: Option<Fluid>,
    /// Fluid level as in vanilla: 0 = source, 1..=7 flowing, 8+ falling.
    pub fluid_level: u8,
    /// Coulomb friction coefficient of this block's surface.
    pub friction: f32,
    /// Bounciness of this block.
    pub restitution: f32,
    /// Density in kg/m³ when this block is used as a body.
    pub density: f32,
}

impl StateInfo {
    /// A state that is empty space.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            boxes: Vec::new(),
            outline: Vec::new(),
            fluid: None,
            fluid_level: 0,
            friction: 0.7,
            restitution: 0.0,
            density: 2000.0,
        }
    }

    /// A full solid cube (used for unloaded chunks).
    #[must_use]
    pub fn solid() -> Self {
        Self {
            boxes: vec![[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
            ..Self::empty()
        }
    }
}

/// Read-only access to the Minecraft world, implemented by the server glue.
pub trait WorldView {
    /// The block state id at `cell`, or `None` if that chunk is not loaded.
    fn state_at(&mut self, cell: Cell) -> Option<u16>;
    /// Describes a block state. Called at most once per state id.
    fn describe(&mut self, state: u16) -> StateInfo;
    /// Lowest valid Y of the world.
    fn min_y(&self) -> i32;
}

/// Tunables for the simulation.
#[derive(Clone, Debug)]
pub struct Config {
    /// Gravity in m/s² (1 block = 1 m).
    pub gravity: f32,
    /// Length of a game tick in seconds.
    pub tick_dt: f32,
    /// Physics sub-steps per game tick (more = more stable stacking / thin colliders).
    pub substeps: u32,
    /// Hard speed cap in m/s, purely a safety net.
    pub max_speed: f32,
    /// Bodies falling this far below the world are deleted.
    pub void_margin: f32,
    /// How often (ticks) cells near awake bodies are re-read from the world.
    pub awake_refresh_ticks: u64,
    /// How often (ticks) cells near sleeping bodies are re-read from the world.
    pub sleep_refresh_ticks: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            tick_dt: 0.05,
            substeps: 4,
            max_speed: 90.0,
            void_margin: 64.0,
            awake_refresh_ticks: 2,
            sleep_refresh_ticks: 10,
        }
    }
}

/// One block of a body.
#[derive(Clone, Debug)]
pub struct Part {
    /// Offset in whole blocks from the body origin (the origin is the centre of the offset-(0,0,0) block).
    pub offset: [i32; 3],
    /// Collision boxes in block-local coordinates (0..1).
    pub boxes: Vec<[f32; 6]>,
    /// Density in kg/m³.
    pub density: f32,
    /// Friction coefficient.
    pub friction: f32,
    /// Restitution coefficient.
    pub restitution: f32,
}

/// Physical description of a body to spawn: one or more rigidly connected blocks.
#[derive(Clone, Debug)]
pub struct BodySpec {
    /// The blocks.
    pub parts: Vec<Part>,
}

impl BodySpec {
    /// A body made of a single block.
    #[must_use]
    pub fn single(boxes: Vec<[f32; 6]>, density: f32, friction: f32, restitution: f32) -> Self {
        Self { parts: vec![Part { offset: [0, 0, 0], boxes, density, friction, restitution }] }
    }
}

/// A body that was split off another one when a block was removed.
#[derive(Clone, Debug)]
pub struct Split {
    /// The new body.
    pub body: BodyId,
    /// Ids of the parts that now belong to it (they keep their ids and offsets).
    pub parts: Vec<u32>,
}

/// Result of removing a block from a body.
#[derive(Clone, Debug)]
pub enum Removed {
    /// No such body/part.
    NotFound,
    /// That was the last block; the body no longer exists.
    BodyGone,
    /// Removed; disconnected pieces (possibly none) became new bodies.
    Ok(Vec<Split>),
}

/// Result of a ray cast against bodies.
#[derive(Clone, Debug)]
pub struct Hit {
    /// Body hit.
    pub body: BodyId,
    /// Part (block) hit.
    pub part: u32,
    /// Distance along the ray.
    pub toi: f32,
    /// World hit point.
    pub point: [f64; 3],
    /// Face normal in the body's own (unrotated) frame as a unit axis, e.g. `[0, 1, 0]` = top face.
    pub normal_local: [i32; 3],
}

/// A player (or other entity) that can shove bodies around.
#[derive(Clone, Copy, Debug)]
pub struct PusherInput {
    /// Unique key, e.g. the entity id.
    pub key: u64,
    /// Feet position in world coordinates.
    pub feet: [f64; 3],
    /// Half of the entity width.
    pub half_width: f32,
    /// Entity height.
    pub height: f32,
}

/// Where a body currently is.
#[derive(Clone, Copy, Debug)]
pub struct BodyUpdate {
    /// Which body.
    pub id: BodyId,
    /// World position of the body origin (the block centre).
    pub pos: [f64; 3],
    /// Orientation quaternion `[x, y, z, w]`.
    pub rot: [f32; 4],
    /// Whether the body is now asleep (a final update is sent when it falls asleep).
    pub asleep: bool,
    /// The body left the world and has been removed from the engine.
    pub removed: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Content {
    Unloaded,
    State(u16),
}

struct CellEntry {
    content: Content,
    info: Arc<StateInfo>,
    colliders: Vec<ColliderHandle>,
    checked: u64,
    used: u64,
}

struct PartRt {
    id: u32,
    offset: [i32; 3],
    colliders: Vec<ColliderHandle>,
    /// Buoyancy sample points: (local position, volume in m³).
    samples: Vec<(Vector, f32)>,
    volume: f32,
    boxes: Vec<[f32; 6]>,
    density: f32,
    friction: f32,
    restitution: f32,
}

struct Body {
    handle: RigidBodyHandle,
    parts: Vec<PartRt>,
    next_part: u32,
    last_pos: Vector,
    last_rot: Rotation,
    never_sent: bool,
    sent_asleep: bool,
}

struct Pusher {
    handle: RigidBodyHandle,
    last_feet: Vector,
}

/// The physics world for one Minecraft dimension.
pub struct Engine {
    cfg: Config,
    world: PhysicsWorld,
    terrain: RigidBodyHandle,
    cells: HashMap<Cell, CellEntry>,
    infos: HashMap<u16, Arc<StateInfo>>,
    bodies: HashMap<BodyId, Body>,
    pushers: HashMap<u64, Pusher>,
    next_id: BodyId,
    tick: u64,
    /// Floating origin: engine coordinates = world coordinates - origin (keeps f32 precise).
    origin: Option<[i32; 3]>,
    fluid_cells: usize,
    had_fluid: bool,
}

fn v3(a: [f32; 3]) -> Vector {
    Vector::new(a[0], a[1], a[2])
}

impl Engine {
    /// Creates an empty engine.
    #[must_use]
    pub fn new(cfg: Config) -> Self {
        let mut world = PhysicsWorld::default();
        world.gravity = Vector::new(0.0, -cfg.gravity, 0.0);
        world.integration_parameters.dt = cfg.tick_dt / cfg.substeps as f32;
        world.integration_parameters.num_solver_iterations = 8;
        // Wide speculative-contact margin: with rapier's tiny default, fast impacts lose their
        // restitution (a 10 m/s slime-block drop only came back at 1 m/s).
        world.integration_parameters.normalized_prediction_distance = 0.002;
        let terrain = world.insert_body(RigidBodyBuilder::fixed());
        Self {
            cfg,
            world,
            terrain,
            cells: HashMap::new(),
            infos: HashMap::new(),
            bodies: HashMap::new(),
            pushers: HashMap::new(),
            next_id: 1,
            tick: 0,
            origin: None,
            fluid_cells: 0,
            had_fluid: false,
        }
    }

    /// Number of live bodies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// Whether there are no bodies.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// All body ids.
    #[must_use]
    pub fn ids(&self) -> Vec<BodyId> {
        self.bodies.keys().copied().collect()
    }

    /// Whether the body exists.
    #[must_use]
    pub fn contains(&self, id: BodyId) -> bool {
        self.bodies.contains_key(&id)
    }

    // ------------------------------------------------------------------
    // coordinate helpers
    // ------------------------------------------------------------------

    fn origin(&self) -> [i32; 3] {
        self.origin.unwrap_or([0, 0, 0])
    }

    fn to_local(&self, p: [f64; 3]) -> Vector {
        let o = self.origin();
        Vector::new(
            (p[0] - f64::from(o[0])) as f32,
            (p[1] - f64::from(o[1])) as f32,
            (p[2] - f64::from(o[2])) as f32,
        )
    }

    fn to_world(&self, p: Vector) -> [f64; 3] {
        let o = self.origin();
        [
            f64::from(p.x) + f64::from(o[0]),
            f64::from(p.y) + f64::from(o[1]),
            f64::from(p.z) + f64::from(o[2]),
        ]
    }

    fn cell_to_local(&self, c: Cell) -> Vector {
        let o = self.origin();
        Vector::new(
            (c[0] - o[0]) as f32,
            (c[1] - o[1]) as f32,
            (c[2] - o[2]) as f32,
        )
    }

    fn local_to_cell(&self, p: Vector) -> Cell {
        let o = self.origin();
        [
            p.x.floor() as i32 + o[0],
            p.y.floor() as i32 + o[1],
            p.z.floor() as i32 + o[2],
        ]
    }

    // ------------------------------------------------------------------
    // state info cache
    // ------------------------------------------------------------------

    /// Returns (and caches) the description of a block state.
    pub fn info(&mut self, view: &mut dyn WorldView, state: u16) -> Arc<StateInfo> {
        if let Some(i) = self.infos.get(&state) {
            return Arc::clone(i);
        }
        let info = Arc::new(view.describe(state));
        self.infos.insert(state, Arc::clone(&info));
        info
    }

    // ------------------------------------------------------------------
    // bodies
    // ------------------------------------------------------------------

    fn build_part(world: &mut PhysicsWorld, handle: RigidBodyHandle, id: u32, part: &Part, n: i32) -> PartRt {
        let mut colliders = Vec::new();
        let mut samples = Vec::new();
        let mut volume = 0.0;
        let off = Vector::new(part.offset[0] as f32, part.offset[1] as f32, part.offset[2] as f32);
        for b in &part.boxes {
            let half = [(b[3] - b[0]) * 0.5, (b[4] - b[1]) * 0.5, (b[5] - b[2]) * 0.5];
            if half.iter().any(|h| *h <= 1e-4) {
                continue;
            }
            let center = Vector::new((b[0] + b[3]) * 0.5 - 0.5, (b[1] + b[4]) * 0.5 - 0.5, (b[2] + b[5]) * 0.5 - 0.5) + off;
            let collider = ColliderBuilder::cuboid(half[0], half[1], half[2])
                .translation(center)
                .density(part.density)
                .friction(part.friction)
                .restitution(part.restitution)
                .friction_combine_rule(CoefficientCombineRule::Min)
                .restitution_combine_rule(CoefficientCombineRule::Max)
                .build();
            colliders.push(world.insert_collider(collider, Some(handle)));

            let vol = (half[0] * 2.0) * (half[1] * 2.0) * (half[2] * 2.0);
            volume += vol;
            let each = vol / (n * n * n) as f32;
            for ix in 0..n {
                for iy in 0..n {
                    for iz in 0..n {
                        let f = |i: i32, h: f32, c: f32| c - h + (i as f32 + 0.5) / n as f32 * h * 2.0;
                        samples.push((Vector::new(f(ix, half[0], center.x), f(iy, half[1], center.y), f(iz, half[2], center.z)), each));
                    }
                }
            }
        }
        PartRt {
            id,
            offset: part.offset,
            colliders,
            samples,
            volume,
            boxes: part.boxes.clone(),
            density: part.density,
            friction: part.friction,
            restitution: part.restitution,
        }
    }

    fn new_dynamic(&mut self, pose: Pose) -> RigidBodyHandle {
        let mut rb = RigidBodyBuilder::dynamic()
            .pose(pose)
            .ccd_enabled(true)
            .linear_damping(0.005)
            .angular_damping(0.03)
            .build();
        rb.activation_mut().normalized_linear_threshold = 0.06;
        rb.activation_mut().angular_threshold = 0.06;
        rb.activation_mut().time_until_sleep = 1.0;
        self.world.insert_body(rb)
    }

    /// Spawns a body whose origin (centre of the offset-0 block) is at `pos`.
    /// Parts get ids `0..n` in spec order.
    pub fn spawn(&mut self, spec: &BodySpec, pos: [f64; 3]) -> BodyId {
        if self.origin.is_none() {
            // Round to a multiple of 16 so the origin is stable and tidy.
            self.origin = Some([
                (pos[0].floor() as i32).div_euclid(16) * 16,
                (pos[1].floor() as i32).div_euclid(16) * 16,
                (pos[2].floor() as i32).div_euclid(16) * 16,
            ]);
        }
        let p = self.to_local(pos);
        let handle = self.new_dynamic(Pose::from_translation(p));
        let n = if spec.parts.len() > 2 { 2 } else { 3 };
        let parts: Vec<PartRt> = spec.parts.iter().enumerate().map(|(i, part)| Self::build_part(&mut self.world, handle, i as u32, part, n)).collect();
        let next_part = parts.len() as u32;
        let id = self.next_id;
        self.next_id += 1;
        self.bodies.insert(
            id,
            Body { handle, parts, next_part, last_pos: p, last_rot: Rotation::IDENTITY, never_sent: true, sent_asleep: false },
        );
        id
    }

    /// The parts of a body as `(part id, offset)`.
    #[must_use]
    pub fn parts_of(&self, id: BodyId) -> Vec<(u32, [i32; 3])> {
        self.bodies.get(&id).map_or_else(Vec::new, |b| b.parts.iter().map(|p| (p.id, p.offset)).collect())
    }

    /// Adds a block to a body (wakes it). Returns the new part id, or `None` if that offset is taken.
    pub fn add_part(&mut self, id: BodyId, part: &Part) -> Option<u32> {
        let b = self.bodies.get_mut(&id)?;
        if b.parts.iter().any(|p| p.offset == part.offset) {
            return None;
        }
        let pid = b.next_part;
        b.next_part += 1;
        let n = if b.parts.len() + 1 > 2 { 2 } else { 3 };
        let rt = Self::build_part(&mut self.world, b.handle, pid, part, n);
        b.parts.push(rt);
        let h = b.handle;
        self.world.wake_up(h, true);
        Some(pid)
    }

    /// Removes one block from a body. If the rest is no longer connected, each disconnected
    /// piece keeps flying as its own body with the correct velocity.
    pub fn remove_part(&mut self, id: BodyId, part_id: u32) -> Removed {
        let Some(b) = self.bodies.get_mut(&id) else { return Removed::NotFound };
        let Some(idx) = b.parts.iter().position(|p| p.id == part_id) else { return Removed::NotFound };
        let gone = b.parts.remove(idx);
        for c in gone.colliders {
            self.world.remove_collider(c);
        }
        if b.parts.is_empty() {
            self.remove(id);
            return Removed::BodyGone;
        }
        let handle = b.handle;
        self.world.wake_up(handle, true);

        // Connected components over face adjacency.
        let offsets: HashMap<[i32; 3], usize> = b.parts.iter().enumerate().map(|(i, p)| (p.offset, i)).collect();
        let mut comp = vec![usize::MAX; b.parts.len()];
        let mut ncomp = 0;
        for start in 0..b.parts.len() {
            if comp[start] != usize::MAX {
                continue;
            }
            let mut stack = vec![start];
            comp[start] = ncomp;
            while let Some(i) = stack.pop() {
                let o = b.parts[i].offset;
                for d in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] {
                    if let Some(&j) = offsets.get(&[o[0] + d[0], o[1] + d[1], o[2] + d[2]]) {
                        if comp[j] == usize::MAX {
                            comp[j] = ncomp;
                            stack.push(j);
                        }
                    }
                }
            }
            ncomp += 1;
        }
        if ncomp <= 1 {
            return Removed::Ok(Vec::new());
        }

        // Keep the biggest component in this body, split the others off.
        let mut sizes = vec![0usize; ncomp];
        for c in &comp {
            sizes[*c] += 1;
        }
        let keep = (0..ncomp).max_by_key(|c| sizes[*c]).unwrap_or(0);
        let mut moving: Vec<Vec<PartRt>> = (0..ncomp).map(|_| Vec::new()).collect();
        let old_parts = std::mem::take(&mut b.parts);
        let mut kept = Vec::new();
        for (p, c) in old_parts.into_iter().zip(comp) {
            if c == keep {
                kept.push(p);
            } else {
                moving[c].push(p);
            }
        }
        let n = if kept.len() > 2 { 2 } else { 3 };
        let _ = n;
        b.parts = kept;

        let src_pose = *self.world.bodies[handle].position();
        let src_ang = self.world.bodies[handle].angvel();
        let src_rot = *self.world.bodies[handle].rotation();
        let src_body = self.world.bodies[handle].clone();
        let mut splits = Vec::new();
        for group in moving.into_iter().filter(|g| !g.is_empty()) {
            // Re-create the group's colliders on a fresh body with the same pose.
            let new_handle = self.new_dynamic(src_pose);
            let mut parts = Vec::new();
            let mut ids = Vec::new();
            let n = if group.len() > 2 { 2 } else { 3 };
            for old in group {
                for c in &old.colliders {
                    self.world.remove_collider(*c);
                }
                let spec = Part { offset: old.offset, boxes: old.boxes.clone(), density: old.density, friction: old.friction, restitution: old.restitution };
                let rt = Self::build_part(&mut self.world, new_handle, old.id, &spec, n);
                ids.push(rt.id);
                parts.push(rt);
            }
            let com = self.world.bodies[new_handle].center_of_mass();
            let lin = src_body.velocity_at_point(com);
            let rb = &mut self.world.bodies[new_handle];
            rb.set_linvel(lin, true);
            rb.set_angvel(src_ang, true);
            let next_part = parts.iter().map(|p| p.id).max().unwrap_or(0) + 1;
            let nid = self.next_id;
            self.next_id += 1;
            self.bodies.insert(
                nid,
                Body { handle: new_handle, parts, next_part, last_pos: src_pose.translation, last_rot: src_rot, never_sent: true, sent_asleep: false },
            );
            splits.push(Split { body: nid, parts: ids });
        }
        Removed::Ok(splits)
    }

    /// World-space bounding box `(min, max)` of a body.
    #[must_use]
    pub fn body_bounds(&self, id: BodyId) -> Option<([f64; 3], [f64; 3])> {
        let b = self.bodies.get(&id)?;
        let (lo, hi) = self.body_aabb(b)?;
        Some((self.to_world(lo), self.to_world(hi)))
    }

    /// Removes a body.
    pub fn remove(&mut self, id: BodyId) {
        if let Some(b) = self.bodies.remove(&id) {
            self.world.remove_body_with_colliders(b.handle, true);
        }
    }

    /// Removes every body (terrain cache and pushers are dropped too).
    pub fn clear(&mut self) {
        let ids = self.ids();
        for id in ids {
            self.remove(id);
        }
        let keys: Vec<u64> = self.pushers.keys().copied().collect();
        for k in keys {
            self.remove_pusher(k);
        }
        let cells: Vec<Cell> = self.cells.keys().copied().collect();
        for c in cells {
            self.drop_cell(c);
        }
    }

    /// Sets a body's linear velocity in m/s (world axes). Mostly useful for tests and spawn velocity.
    pub fn set_velocity(&mut self, id: BodyId, lin: [f32; 3]) {
        if let Some(b) = self.bodies.get(&id) {
            self.world.bodies[b.handle].set_linvel(v3(lin), true);
        }
    }

    /// Converts a world point on a body into the body's local frame (for [`Engine::drag`]).
    #[must_use]
    pub fn grab_point_local(&self, id: BodyId, point: [f64; 3]) -> Option<[f32; 3]> {
        let b = self.bodies.get(&id)?;
        let rb = &self.world.bodies[b.handle];
        let l = rb.rotation().inverse() * (self.to_local(point) - rb.translation());
        Some([l.x, l.y, l.z])
    }

    /// Gently pulls the point `grab` (body-local) towards `target` (world). The pull is applied as an
    /// impulse *at that point*, so grabbing a block off-centre makes it swing and rotate naturally.
    /// `gain` is 1/s (speed per metre of error), `max_speed` caps the pull speed in m/s and `blend`
    /// (0..1) is how much of the speed error is corrected each tick (lower = weaker / floatier).
    /// Gravity is cancelled while held and the spin is damped a little. Returns false if the body is gone.
    pub fn drag(&mut self, id: BodyId, grab: [f32; 3], target: [f64; 3], gain: f32, max_speed: f32, blend: f32) -> bool {
        let Some(b) = self.bodies.get(&id) else { return false };
        let handle = b.handle;
        let tgt = self.to_local(target);
        let (g, dt) = (self.cfg.gravity, self.cfg.tick_dt);
        let rb = &mut self.world.bodies[handle];
        let p = rb.translation() + *rb.rotation() * v3(grab);
        let com = rb.center_of_mass();
        let vg = rb.linvel() + rb.angvel().cross(p - com);
        let mut want = (tgt - p) * gain;
        let len = want.length();
        if len > max_speed {
            want *= max_speed / len;
        }
        let m = rb.mass();
        rb.apply_impulse_at_point((want - vg) * (blend * m), p, true);
        rb.apply_impulse(Vector::new(0.0, g * m * dt, 0.0), true);
        let w = rb.angvel();
        rb.set_angvel(w * 0.9, true);
        true
    }

    /// Sets a body's orientation (quaternion `[x, y, z, w]`), e.g. when restoring a saved body.
    pub fn set_rotation(&mut self, id: BodyId, rot: [f32; 4]) {
        if let Some(b) = self.bodies.get(&id) {
            let q = Rotation::from_xyzw(rot[0], rot[1], rot[2], rot[3]).normalize();
            self.world.bodies[b.handle].set_rotation(q, true);
        }
    }

    /// Current state of a body: world position, rotation, asleep.
    #[must_use]
    pub fn body_state(&self, id: BodyId) -> Option<([f64; 3], [f32; 4], bool)> {
        let b = self.bodies.get(&id)?;
        let rb = &self.world.bodies[b.handle];
        let r = rb.rotation();
        Some((self.to_world(rb.translation()), [r.x, r.y, r.z, r.w], rb.is_sleeping()))
    }

    /// Current linear velocity of a body in m/s.
    #[must_use]
    pub fn velocity(&self, id: BodyId) -> Option<[f32; 3]> {
        let b = self.bodies.get(&id)?;
        let v = self.world.bodies[b.handle].linvel();
        Some([v.x, v.y, v.z])
    }

    /// Mass of a body in kg.
    #[must_use]
    pub fn mass(&self, id: BodyId) -> Option<f32> {
        let b = self.bodies.get(&id)?;
        Some(self.world.bodies[b.handle].mass())
    }

    /// Finds the block of a body hit first by a ray (e.g. a player's line of sight).
    #[must_use]
    pub fn raycast(&self, origin: [f64; 3], dir: [f32; 3], max_dist: f32) -> Option<Hit> {
        let o = self.to_local(origin);
        let d = v3(dir).normalize_or_zero();
        if d == Vector::ZERO {
            return None;
        }
        let ray = Ray::new(o, d);
        let mut best: Option<(BodyId, u32, f32, Vector)> = None;
        for (id, b) in &self.bodies {
            for part in &b.parts {
                for ch in &part.colliders {
                    let Some(col) = self.world.colliders.get(*ch) else { continue };
                    if let Some(hit) = col.shape().cast_ray_and_get_normal(col.position(), &ray, max_dist, true) {
                        if best.as_ref().is_none_or(|(_, _, t, _)| hit.time_of_impact < *t) {
                            best = Some((*id, part.id, hit.time_of_impact, hit.normal));
                        }
                    }
                }
            }
        }
        let (body, part, toi, normal) = best?;
        let rot = *self.world.bodies[self.bodies[&body].handle].rotation();
        let n = rot.inverse() * normal;
        let a = n.abs();
        let local = if a.x >= a.y && a.x >= a.z {
            [n.x.signum() as i32, 0, 0]
        } else if a.y >= a.z {
            [0, n.y.signum() as i32, 0]
        } else {
            [0, 0, n.z.signum() as i32]
        };
        Some(Hit { body, part, toi, point: self.to_world(o + d * toi), normal_local: local })
    }

    /// Hits a body with an impulse (kg·m/s) along `dir`, applied where the ray from `origin`
    /// touches the body so off-centre hits make it spin. The resulting speed change is
    /// capped at `max_dv` m/s.
    pub fn punch(&mut self, id: BodyId, origin: [f64; 3], dir: [f32; 3], impulse: f32, max_dv: f32) -> bool {
        let o = self.to_local(origin);
        let d = v3(dir).normalize_or_zero();
        if d == Vector::ZERO {
            return false;
        }
        let Some(b) = self.bodies.get(&id) else { return false };
        let ray = Ray::new(o, d);
        let mut toi_best: Option<f32> = None;
        for ch in b.parts.iter().flat_map(|p| p.colliders.iter()) {
            if let Some(col) = self.world.colliders.get(*ch) {
                if let Some(t) = col.shape().cast_ray(col.position(), &ray, 16.0, true) {
                    if toi_best.is_none_or(|bt| t < bt) {
                        toi_best = Some(t);
                    }
                }
            }
        }
        let rb = &mut self.world.bodies[b.handle];
        let point = toi_best.map_or_else(|| rb.center_of_mass(), |t| o + d * t);
        let mass = rb.mass().max(1.0);
        let impulse = impulse.min(mass * max_dv);
        rb.apply_impulse_at_point(d * impulse, point, true);
        true
    }

    // ------------------------------------------------------------------
    // pushers (players)
    // ------------------------------------------------------------------

    fn remove_pusher(&mut self, key: u64) {
        if let Some(p) = self.pushers.remove(&key) {
            self.world.remove_body_with_colliders(p.handle, true);
        }
    }

    fn update_pushers(&mut self, inputs: &[PusherInput]) {
        let present: HashSet<u64> = inputs.iter().map(|p| p.key).collect();
        let stale: Vec<u64> = self.pushers.keys().filter(|k| !present.contains(k)).copied().collect();
        for k in stale {
            self.remove_pusher(k);
        }
        if self.bodies.is_empty() {
            let all: Vec<u64> = self.pushers.keys().copied().collect();
            for k in all {
                self.remove_pusher(k);
            }
            return;
        }
        for input in inputs {
            let feet = self.to_local(input.feet);
            let near = self.bodies.values().any(|b| {
                let t = self.world.bodies[b.handle].translation();
                (t - feet).length_squared() < 6.0 * 6.0
            });
            if !near {
                self.remove_pusher(input.key);
                continue;
            }
            match self.pushers.get_mut(&input.key) {
                Some(p) => {
                    let rb = &mut self.world.bodies[p.handle];
                    if (feet - p.last_feet).length() > 3.0 {
                        // Teleport: move without imparting velocity.
                        rb.set_translation(feet, false);
                    } else {
                        rb.set_next_kinematic_translation(feet);
                    }
                    p.last_feet = feet;
                }
                None => {
                    let rb = RigidBodyBuilder::kinematic_position_based().translation(feet).build();
                    let handle = self.world.insert_body(rb);
                    let collider = ColliderBuilder::cuboid(input.half_width, input.height * 0.5, input.half_width)
                        .translation(Vector::new(0.0, input.height * 0.5, 0.0))
                        .friction(0.0)
                        .build();
                    self.world.insert_collider(collider, Some(handle));
                    self.pushers.insert(input.key, Pusher { handle, last_feet: feet });
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // terrain
    // ------------------------------------------------------------------

    fn drop_cell(&mut self, cell: Cell) {
        if let Some(entry) = self.cells.remove(&cell) {
            if entry.info.fluid.is_some() {
                self.fluid_cells -= 1;
            }
            for c in entry.colliders {
                self.world.remove_collider(c);
            }
        }
    }

    /// Re-reads one cell from the world and rebuilds its colliders if it changed.
    /// Returns true if an *existing* cell changed content.
    fn refresh_cell(&mut self, view: &mut dyn WorldView, cell: Cell) -> bool {
        let content = view.state_at(cell).map_or(Content::Unloaded, Content::State);
        if let Some(entry) = self.cells.get_mut(&cell) {
            entry.checked = self.tick;
            entry.used = self.tick;
            if entry.content == content {
                return false;
            }
        }
        let info = match content {
            Content::Unloaded => Arc::new(StateInfo::solid()),
            Content::State(s) => self.info(view, s),
        };
        let existed = self.cells.contains_key(&cell);
        if existed {
            self.drop_cell(cell);
        }

        let base = self.cell_to_local(cell);
        let mut colliders = Vec::new();
        for b in &info.boxes {
            let half = Vector::new((b[3] - b[0]) * 0.5, (b[4] - b[1]) * 0.5, (b[5] - b[2]) * 0.5);
            if half.x <= 1e-4 || half.y <= 1e-4 || half.z <= 1e-4 {
                continue;
            }
            let center = base + Vector::new((b[0] + b[3]) * 0.5, (b[1] + b[4]) * 0.5, (b[2] + b[5]) * 0.5);
            let collider = ColliderBuilder::cuboid(half.x, half.y, half.z)
                .translation(center)
                .friction(info.friction)
                .restitution(info.restitution)
                .friction_combine_rule(CoefficientCombineRule::Min)
                .restitution_combine_rule(CoefficientCombineRule::Max)
                .build();
            colliders.push(self.world.insert_collider(collider, Some(self.terrain)));
        }
        if info.fluid.is_some() {
            self.fluid_cells += 1;
        }
        self.cells.insert(
            cell,
            CellEntry { content, info, colliders, checked: self.tick, used: self.tick },
        );
        existed
    }

    fn body_aabb(&self, b: &Body) -> Option<(Vector, Vector)> {
        let mut min = Vector::splat(f32::MAX);
        let mut max = Vector::splat(f32::MIN);
        let mut any = false;
        for ch in b.parts.iter().flat_map(|p| p.colliders.iter()) {
            if let Some(c) = self.world.colliders.get(*ch) {
                let a = c.compute_aabb();
                min = min.min(a.mins);
                max = max.max(a.maxs);
                any = true;
            }
        }
        any.then_some((min, max))
    }

    fn refresh_terrain(&mut self, view: &mut dyn WorldView) {
        let t = self.tick;
        let dt = self.cfg.tick_dt;
        let mut changed: Vec<Cell> = Vec::new();

        // Sleeping bodies only look at their surroundings once per refresh interval (spread over the ticks
        // by body id), so a sleeping body costs neither the AABB nor the per-cell lookups on other ticks.
        // Cells are only ever re-read once they are `interval` ticks stale anyway, and the 200-tick
        // eviction below is far longer than the interval, so nothing is dropped early.
        let sleep_every = self.cfg.sleep_refresh_ticks.max(1);
        let handles: Vec<(RigidBodyHandle, Option<(Vector, Vector)>)> = self
            .bodies
            .iter()
            .filter_map(|(id, b)| {
                if self.world.bodies[b.handle].is_sleeping() && t.wrapping_add(*id) % sleep_every != 0 {
                    return None;
                }
                Some((b.handle, self.body_aabb(b)))
            })
            .collect();

        for (h, aabb) in handles {
            let Some((min, max)) = aabb else { continue };
            let rb = &self.world.bodies[h];
            let asleep = rb.is_sleeping();
            let speed = rb.linvel().length();
            let margin = 0.3 + speed * dt;
            let lo = self.local_to_cell(min - Vector::splat(margin));
            let hi = self.local_to_cell(max + Vector::splat(margin));
            let interval = if asleep { self.cfg.sleep_refresh_ticks } else { self.cfg.awake_refresh_ticks };
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        let cell = [x, y, z];
                        let stale = match self.cells.get_mut(&cell) {
                            None => true,
                            Some(e) => {
                                e.used = t;
                                t.saturating_sub(e.checked) >= interval
                            }
                        };
                        if stale && self.refresh_cell(view, cell) {
                            changed.push(cell);
                        }
                    }
                }
            }
        }

        // Wake every body that touches a cell which changed under it.
        if !changed.is_empty() {
            let wake: Vec<RigidBodyHandle> = self
                .bodies
                .values()
                .filter_map(|b| {
                    let (min, max) = self.body_aabb(b)?;
                    let touched = changed.iter().any(|c| {
                        let lo = self.cell_to_local(*c) - Vector::splat(0.5);
                        let hi = lo + Vector::splat(2.0);
                        min.x <= hi.x && max.x >= lo.x && min.y <= hi.y && max.y >= lo.y && min.z <= hi.z && max.z >= lo.z
                    });
                    touched.then_some(b.handle)
                })
                .collect();
            for h in wake {
                self.world.wake_up(h, true);
            }
        }

        // Drop cells nobody has needed for a while.
        if t % 100 == 0 {
            let stale: Vec<Cell> = self.cells.iter().filter(|(_, e)| t.saturating_sub(e.used) > 200).map(|(c, _)| *c).collect();
            for c in stale {
                self.drop_cell(c);
            }
        }
    }

    // ------------------------------------------------------------------
    // fluids
    // ------------------------------------------------------------------

    /// If the world point is inside fluid, returns the fluid.
    fn fluid_at(&self, p: Vector) -> Option<Fluid> {
        let cell = self.local_to_cell(p);
        let e = self.cells.get(&cell)?;
        let fluid = e.info.fluid?;
        let above = [cell[0], cell[1] + 1, cell[2]];
        let full = self.cells.get(&above).is_some_and(|a| a.info.fluid == Some(fluid));
        let height = if full || e.info.fluid_level >= 8 {
            1.0
        } else {
            (8.0 - f32::from(e.info.fluid_level)) / 9.0
        };
        let base = self.cell_to_local(cell).y;
        (p.y < base + height).then_some(fluid)
    }

    /// Applies buoyancy for the coming sub-step and returns, per body, the submerged
    /// volume fraction and dominant fluid (used for drag afterwards).
    fn apply_fluid_forces(&mut self) -> Vec<(RigidBodyHandle, f32, Fluid)> {
        if self.fluid_cells == 0 {
            if self.had_fluid {
                self.had_fluid = false;
                let hs: Vec<_> = self.bodies.values().map(|b| b.handle).collect();
                for h in hs {
                    self.world.bodies[h].reset_forces(false);
                }
            }
            return Vec::new();
        }
        self.had_fluid = true;
        let g = self.cfg.gravity;
        let mut out = Vec::new();
        let mut jobs: Vec<(RigidBodyHandle, f32, Vec<(Vector, f32, Fluid)>)> = Vec::new();

        for b in self.bodies.values() {
            let rb = &self.world.bodies[b.handle];
            if rb.is_sleeping() {
                continue;
            }
            let pose = *rb.position();
            let mut wet = Vec::new();
            let mut total = 0.0;
            for part in &b.parts {
                total += part.volume;
                for (local, vol) in &part.samples {
                    let p = pose * *local;
                    if let Some(f) = self.fluid_at(p) {
                        wet.push((p, *vol, f));
                    }
                }
            }
            jobs.push((b.handle, total.max(1e-6), wet));
        }

        for (h, total, wet) in jobs {
            let rb = &mut self.world.bodies[h];
            rb.reset_forces(false);
            if wet.is_empty() {
                continue;
            }
            let mut water = 0.0;
            let mut lava = 0.0;
            for (p, vol, f) in &wet {
                let rho = match f {
                    Fluid::Water => {
                        water += vol;
                        WATER_DENSITY
                    }
                    Fluid::Lava => {
                        lava += vol;
                        LAVA_DENSITY
                    }
                };
                rb.add_force_at_point(Vector::new(0.0, rho * g * vol, 0.0), *p, true);
            }
            let fluid = if lava > water { Fluid::Lava } else { Fluid::Water };
            out.push((h, ((water + lava) / total).min(1.0), fluid));
        }
        out
    }

    fn apply_fluid_drag(&mut self, wet: &[(RigidBodyHandle, f32, Fluid)]) {
        let dt = self.world.integration_parameters.dt;
        for (h, frac, fluid) in wet {
            // Viscous drag as an exponential velocity decay (unconditionally stable).
            let (k_lin, k_ang) = match fluid {
                Fluid::Water => (1.6, 2.5),
                Fluid::Lava => (6.0, 8.0),
            };
            let rb = &mut self.world.bodies[*h];
            let lin = rb.linvel() * (-k_lin * frac * dt).exp();
            let ang = rb.angvel() * (-k_ang * frac * dt).exp();
            rb.set_linvel(lin, false);
            rb.set_angvel(ang, false);
        }
    }

    // ------------------------------------------------------------------
    // main loop
    // ------------------------------------------------------------------

    /// Advances the simulation by one game tick and reports bodies whose visuals changed.
    pub fn tick(&mut self, view: &mut dyn WorldView, pushers: &[PusherInput]) -> Vec<BodyUpdate> {
        self.tick += 1;
        if self.bodies.is_empty() {
            return Vec::new();
        }
        self.update_pushers(pushers);
        self.refresh_terrain(view);

        let max_speed = self.cfg.max_speed;
        for _ in 0..self.cfg.substeps {
            let wet = self.apply_fluid_forces();
            self.world.step();
            self.apply_fluid_drag(&wet);
            // Safety net against runaway velocities.
            for b in self.bodies.values() {
                let rb = &mut self.world.bodies[b.handle];
                if !rb.is_sleeping() {
                    let v = rb.linvel();
                    let s = v.length();
                    if s > max_speed {
                        rb.set_linvel(v * (max_speed / s), false);
                    }
                }
            }
        }

        let origin = self.origin();
        let void_y = (view.min_y() as f32) - self.cfg.void_margin - origin[1] as f32;
        let mut updates = Vec::new();
        let mut dead = Vec::new();
        let ids: Vec<BodyId> = self.bodies.keys().copied().collect();
        for id in ids {
            let (handle, never_sent, sent_asleep, last_pos, last_rot) = {
                let b = &self.bodies[&id];
                (b.handle, b.never_sent, b.sent_asleep, b.last_pos, b.last_rot)
            };
            let rb = &self.world.bodies[handle];
            let pos = rb.translation();
            let rot = *rb.rotation();
            let asleep = rb.is_sleeping();

            if pos.y < void_y || !pos.is_finite() {
                dead.push(id);
                continue;
            }
            let moved = (pos - last_pos).length() > 0.0015 || (1.0 - last_rot.dot(rot).abs()) > 1e-7;
            let send = never_sent || moved || (asleep && !sent_asleep);
            if send {
                updates.push(BodyUpdate {
                    id,
                    pos: [
                        f64::from(pos.x) + f64::from(origin[0]),
                        f64::from(pos.y) + f64::from(origin[1]),
                        f64::from(pos.z) + f64::from(origin[2]),
                    ],
                    rot: [rot.x, rot.y, rot.z, rot.w],
                    asleep,
                    removed: false,
                });
                if let Some(b) = self.bodies.get_mut(&id) {
                    b.last_pos = pos;
                    b.last_rot = rot;
                    b.never_sent = false;
                    b.sent_asleep = asleep;
                }
            } else if !asleep {
                if let Some(b) = self.bodies.get_mut(&id) {
                    b.sent_asleep = false;
                }
            }
        }
        for id in dead {
            self.remove(id);
            updates.push(BodyUpdate { id, pos: [0.0; 3], rot: [0.0, 0.0, 0.0, 1.0], asleep: true, removed: true });
        }
        updates
    }
}

// ----------------------------------------------------------------------
// tests (run natively with `cargo test`)
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const AIR: u16 = 0;
    const STONE: u16 = 1;
    const ICE: u16 = 2;
    const SLIME: u16 = 3;
    const WATER: u16 = 4; // source
    const SLAB_TOP: u16 = 5;

    /// A tiny fake world: a sparse map of cells; anything not present is `default`.
    struct Fake {
        cells: HashMap<Cell, u16>,
        default: Option<u16>,
        loaded: Box<dyn Fn(Cell) -> bool>,
    }

    impl Fake {
        fn flat(floor_y: i32, kind: u16) -> Self {
            let mut cells = HashMap::new();
            for x in -40..=80 {
                for z in -40..=40 {
                    cells.insert([x, floor_y, z], kind);
                }
            }
            Self { cells, default: Some(AIR), loaded: Box::new(|_| true) }
        }
    }

    impl WorldView for Fake {
        fn state_at(&mut self, cell: Cell) -> Option<u16> {
            if !(self.loaded)(cell) {
                return None;
            }
            Some(self.cells.get(&cell).copied().or(self.default).unwrap_or(AIR))
        }
        fn describe(&mut self, state: u16) -> StateInfo {
            let full = vec![[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]];
            match state {
                STONE => StateInfo { boxes: full.clone(), outline: full, friction: 0.7, restitution: 0.2, density: 2600.0, ..StateInfo::empty() },
                ICE => StateInfo { boxes: full.clone(), outline: full, friction: 0.035, restitution: 0.1, density: 917.0, ..StateInfo::empty() },
                SLIME => StateInfo { boxes: full.clone(), outline: full, friction: 0.35, restitution: 0.85, density: 1100.0, ..StateInfo::empty() },
                WATER => StateInfo { fluid: Some(Fluid::Water), ..StateInfo::empty() },
                SLAB_TOP => StateInfo { boxes: vec![[0.0, 0.0, 0.0, 1.0, 0.5, 1.0]], outline: vec![[0.0, 0.0, 0.0, 1.0, 0.5, 1.0]], ..StateInfo::empty() },
                _ => StateInfo::empty(),
            }
        }
        fn min_y(&self) -> i32 {
            -64
        }
    }

    fn block(density: f32, friction: f32, restitution: f32) -> BodySpec {
        BodySpec::single(vec![[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], density, friction, restitution)
    }

    fn run(e: &mut Engine, w: &mut Fake, ticks: usize) -> Vec<BodyUpdate> {
        let mut last = Vec::new();
        for _ in 0..ticks {
            last = e.tick(w, &[]);
        }
        last
    }

    fn run_until_asleep(e: &mut Engine, w: &mut Fake, id: BodyId, max_ticks: usize) -> usize {
        for t in 0..max_ticks {
            e.tick(w, &[]);
            if e.body_state(id).is_some_and(|s| s.2) {
                return t;
            }
        }
        max_ticks
    }

    #[test]
    fn falls_lands_flat_and_sleeps() {
        // Stone floor occupies y in [0,1) -> top surface at y = 1.
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&block(1600.0, 0.7, 0.05), [0.5, 11.5, 0.5]);
        let t = run_until_asleep(&mut e, &mut world, id, 400);
        assert!(t < 400, "never fell asleep");
        let (p, r, asleep) = e.body_state(id).unwrap();
        assert!(asleep);
        assert!((p[1] - 1.5).abs() < 0.02, "rest height {}", p[1]);
        assert!((p[0] - 0.5).abs() < 0.05 && (p[2] - 0.5).abs() < 0.05, "drifted: {p:?}");
        assert!(r[3].abs() > 0.999, "should land flat, rot={r:?}");
    }

    #[test]
    fn free_fall_matches_gravity() {
        // No floor nearby: after 1 s the body should have fallen ~ g/2 = 4.9 m (a bit less for air drag).
        let mut world = Fake { cells: HashMap::new(), default: Some(AIR), loaded: Box::new(|_| true) };
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&block(1600.0, 0.7, 0.05), [0.5, 100.5, 0.5]);
        run(&mut e, &mut world, 20);
        let (p, _, _) = e.body_state(id).unwrap();
        let fallen = 100.5 - p[1];
        assert!((fallen - 4.905).abs() < 0.35, "fell {fallen}");
    }

    #[test]
    fn ice_slides_far_stone_does_not() {
        let slide = |floor: u16, mu: f32| {
            let mut world = Fake::flat(0, floor);
            let mut e = Engine::new(Config::default());
            let id = e.spawn(&block(1600.0, mu, 0.05), [0.5, 1.5, 0.5]);
            run(&mut e, &mut world, 10); // settle
            e.set_velocity(id, [5.0, 0.0, 0.0]);
            run(&mut e, &mut world, 200);
            e.body_state(id).unwrap().0[0] - 0.5
        };
        let stone = slide(STONE, 0.7);
        let ice = slide(ICE, 0.035);
        // v^2 / (2 mu g): stone ~1.8 m, ice ~36 m (but capped by the 10 s window / floor size).
        assert!(stone > 1.0 && stone < 3.0, "stone slid {stone}");
        assert!(ice > 15.0, "ice slid {ice}");
    }

    #[test]
    fn slime_bounces_stone_thuds() {
        // Rebound speed right after the first impact, as a fraction of the impact speed.
        let rebound = |floor: u16, restitution: f32| {
            let mut world = Fake::flat(0, floor);
            let mut e = Engine::new(Config::default());
            let id = e.spawn(&block(1600.0, 0.7, restitution), [0.5, 6.5, 0.5]);
            let mut impact = 0.0f32;
            for _ in 0..80 {
                e.tick(&mut world, &[]);
                let vy = e.velocity(id).unwrap()[1];
                if vy < impact {
                    impact = vy;
                } else if impact < -5.0 && vy > impact + 0.5 {
                    return vy.max(0.0) / -impact; // first tick after the velocity reverses
                }
            }
            0.0
        };
        let slime = rebound(SLIME, 0.85);
        let stone = rebound(STONE, 0.05);
        assert!(slime > 0.6, "slime rebounded at {slime} of impact speed");
        assert!(stone < 0.35, "stone rebounded at {stone} of impact speed");
    }

    #[test]
    fn wood_floats_stone_sinks() {
        // Pool: water cells y in 1..=5 over a stone floor at y=0. Surface at 5 + 8/9.
        let mut world = Fake::flat(0, STONE);
        for x in -3..=3 {
            for z in -3..=3 {
                for y in 1..=5 {
                    world.cells.insert([x, y, z], WATER);
                }
            }
        }
        let surface = 5.0 + 8.0 / 9.0;

        let mut e = Engine::new(Config::default());
        let wood = e.spawn(&block(700.0, 0.5, 0.2), [0.5, 9.5, 0.5]);
        let rock = e.spawn(&block(2600.0, 0.5, 0.2), [-1.5, 9.5, 0.5]);
        run(&mut e, &mut world, 400);

        let (pw, _, _) = e.body_state(wood).unwrap();
        let (pr, _, _) = e.body_state(rock).unwrap();
        // Floating cube of density 0.7 sits 0.7 deep: centre = surface - 0.7 + 0.5.
        let expected = surface - 0.2;
        assert!((pw[1] - expected).abs() < 0.12, "wood centre {} expected {}", pw[1], expected);
        assert!((pr[1] - 1.5).abs() < 0.05, "stone should rest on the pool floor, was {}", pr[1]);
    }

    #[test]
    fn stack_stays_put() {
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let ids: Vec<BodyId> = (0..4).map(|i| e.spawn(&block(1600.0, 0.7, 0.05), [0.5, 1.5 + i as f64 * 1.0, 0.5])).collect();
        run(&mut e, &mut world, 300);
        for (i, id) in ids.iter().enumerate() {
            let (p, _, asleep) = e.body_state(*id).unwrap();
            assert!((p[1] - (1.5 + i as f64)).abs() < 0.06, "block {i} at y={}", p[1]);
            assert!((p[0] - 0.5).abs() < 0.05 && (p[2] - 0.5).abs() < 0.05, "block {i} drifted {p:?}");
            assert!(asleep, "block {i} still awake");
        }
    }

    #[test]
    fn half_slab_world_collision_is_exact() {
        // A (bottom) slab fills the lower half of cell y=1, so the surface is at y = 1.5.
        let mut world = Fake::flat(0, STONE);
        world.cells.insert([0, 1, 0], SLAB_TOP);
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&block(1600.0, 0.7, 0.05), [0.5, 5.5, 0.5]);
        run_until_asleep(&mut e, &mut world, id, 400);
        let (p, _, _) = e.body_state(id).unwrap();
        assert!((p[1] - 2.0).abs() < 0.03, "should rest on slab top at 1.5 -> centre 2.0, got {}", p[1]);
    }

    #[test]
    fn bodies_respect_their_own_shape() {
        // A bottom slab body (half height) must rest with its bottom face on the floor.
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let spec = BodySpec::single(vec![[0.0, 0.0, 0.0, 1.0, 0.5, 1.0]], 700.0, 0.5, 0.1);
        let id = e.spawn(&spec, [0.5, 4.5, 0.5]);
        run_until_asleep(&mut e, &mut world, id, 400);
        let (p, _, asleep) = e.body_state(id).unwrap();
        // Origin is the block centre; slab bottom sits at origin-0.5 -> must touch y=1.
        assert!((p[1] - 1.5).abs() < 0.03, "slab origin y={}", p[1]);
        assert!(asleep);
    }

    #[test]
    fn breaking_the_floor_wakes_and_drops_the_block() {
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&block(1600.0, 0.7, 0.05), [0.5, 1.5, 0.5]);
        run(&mut e, &mut world, 120);
        assert!(e.body_state(id).unwrap().2, "should be asleep before the floor is removed");

        // Remove the whole floor beneath it.
        for x in -2..=3 {
            for z in -2..=3 {
                world.cells.insert([x, 0, z], AIR);
            }
        }
        run(&mut e, &mut world, 60);
        let (p, _, _) = e.body_state(id).unwrap();
        assert!(p[1] < 0.0, "block should have fallen, y={}", p[1]);
    }

    #[test]
    fn unloaded_chunks_are_solid() {
        // Below y=1 is "unloaded": the body must not fall through.
        let mut world = Fake { cells: HashMap::new(), default: Some(AIR), loaded: Box::new(|c| c[1] >= 1) };
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&block(1600.0, 0.7, 0.05), [0.5, 5.5, 0.5]);
        run(&mut e, &mut world, 200);
        let (p, _, _) = e.body_state(id).unwrap();
        assert!((p[1] - 1.5).abs() < 0.05, "should rest on the unloaded boundary, y={}", p[1]);
    }

    #[test]
    fn punch_gives_expected_velocity_and_spin_when_off_centre() {
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&block(1600.0, 0.7, 0.05), [0.5, 50.5, 0.5]); // mid-air: no friction effects
        run(&mut e, &mut world, 2);
        let v0 = e.velocity(id).unwrap();
        let mass = e.mass(id).unwrap();
        assert!((mass - 1600.0).abs() < 1.0, "mass {mass}");
        // Centre punch along +x from the -x side.
        assert!(e.punch(id, [-3.0, 50.5, 0.5], [1.0, 0.0, 0.0], 6000.0, 14.0));
        let v1 = e.velocity(id).unwrap();
        let dv = v1[0] - v0[0];
        assert!((dv - 6000.0 / 1600.0).abs() < 0.2, "dv {dv}");
        // Off-centre: aim at the upper part of the face -> spin about z.
        let id2 = e.spawn(&block(1600.0, 0.7, 0.05), [10.5, 50.5, 0.5]);
        run(&mut e, &mut world, 1);
        assert!(e.punch(id2, [7.0, 50.9, 0.5], [1.0, 0.0, 0.0], 6000.0, 14.0));
        run(&mut e, &mut world, 4);
        let (_, r, _) = e.body_state(id2).unwrap();
        assert!(r[2].abs() > 1e-3, "off-centre hit should rotate the block, rot={r:?}");
    }

    #[test]
    fn raycast_finds_the_right_body() {
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let a = e.spawn(&block(1600.0, 0.7, 0.05), [0.5, 50.5, 0.5]);
        let b = e.spawn(&block(1600.0, 0.7, 0.05), [5.5, 50.5, 0.5]);
        run(&mut e, &mut world, 1);
        let hit = e.raycast([-5.0, 50.5, 0.5], [1.0, 0.0, 0.0], 20.0).unwrap();
        assert_eq!(hit.body, a);
        let hit = e.raycast([20.0, 50.5, 0.5], [-1.0, 0.0, 0.0], 20.0).unwrap();
        assert_eq!(hit.body, b);
        assert!(e.raycast([-5.0, 60.0, 0.5], [1.0, 0.0, 0.0], 20.0).is_none());
    }

    #[test]
    fn a_walking_player_shoves_a_block() {
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&block(700.0, 0.4, 0.1), [3.5, 1.5, 0.5]);
        run(&mut e, &mut world, 5);
        // Player walks +x at 4 m/s, starting 2 blocks behind the block.
        let mut x = 1.0f64;
        for _ in 0..40 {
            x += 4.0 * 0.05;
            let p = PusherInput { key: 1, feet: [x, 1.0, 0.5], half_width: 0.3, height: 1.8 };
            e.tick(&mut world, &[p]);
        }
        let (p, _, _) = e.body_state(id).unwrap();
        assert!(p[0] > 4.5, "block should have been pushed, x={}", p[0]);
    }

    #[test]
    fn many_bodies_stay_fast_enough() {
        // 150 awake bodies for 100 ticks. A generous bound that catches accidental O(n^2) blow-ups.
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        for i in 0..150 {
            let x = (i % 10) as f64 * 1.3 + 0.5;
            let z = ((i / 10) % 5) as f64 * 1.3 + 0.5;
            let y = 3.5 + (i / 50) as f64 * 1.5;
            e.spawn(&block(1600.0, 0.7, 0.05), [x, y, z]);
        }
        let start = std::time::Instant::now();
        run(&mut e, &mut world, 100);
        let per_tick = start.elapsed().as_secs_f64() * 1000.0 / 100.0;
        eprintln!("150 bodies: {per_tick:.2} ms/tick (native, test profile)");
        assert!(per_tick < 500.0, "too slow: {per_tick} ms/tick");
    }

    fn row(n: i32) -> BodySpec {
        BodySpec { parts: (0..n).map(|i| Part { offset: [i, 0, 0], boxes: vec![[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], density: 1600.0, friction: 0.7, restitution: 0.05 }).collect() }
    }

    #[test]
    fn multi_block_body_lands_flat_as_one_rigid_object() {
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&row(3), [0.5, 6.5, 0.5]);
        assert!((e.mass(id).unwrap() - 4800.0).abs() < 1.0);
        let t = run_until_asleep(&mut e, &mut world, id, 500);
        assert!(t < 500);
        let (p, r, _) = e.body_state(id).unwrap();
        assert!((p[1] - 1.5).abs() < 0.03, "y={}", p[1]);
        assert!(r[3].abs() > 0.999, "rot={r:?}");
    }

    #[test]
    fn removing_the_middle_block_splits_the_body() {
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&row(3), [0.5, 40.5, 0.5]);
        run(&mut e, &mut world, 3);
        let Removed::Ok(splits) = e.remove_part(id, 1) else { panic!("expected Ok") };
        assert_eq!(splits.len(), 1);
        assert_eq!(e.parts_of(id).len(), 1);
        assert_eq!(e.parts_of(splits[0].body).len(), 1);
        assert_eq!(e.len(), 2);
        run(&mut e, &mut world, 400);
        for b in [id, splits[0].body] {
            let (p, _, asleep) = e.body_state(b).unwrap();
            assert!(asleep && p[1] < 3.0, "body {b} y={}", p[1]);
        }
        let last = e.parts_of(id)[0].0; // the surviving part may not be id 0 after a split
        assert!(matches!(e.remove_part(id, last), Removed::BodyGone));
    }

    #[test]
    fn can_add_a_block_and_raycast_reports_the_face() {
        let mut world = Fake::flat(0, STONE);
        let mut e = Engine::new(Config::default());
        let id = e.spawn(&block(1600.0, 0.7, 0.05), [0.5, 50.5, 0.5]);
        run(&mut e, &mut world, 1);
        let hit = e.raycast([-5.0, 50.5, 0.5], [1.0, 0.0, 0.0], 20.0).unwrap();
        assert_eq!(hit.normal_local, [-1, 0, 0]);
        let part = Part { offset: [-1, 0, 0], boxes: vec![[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], density: 1600.0, friction: 0.7, restitution: 0.05 };
        let pid = e.add_part(id, &part).unwrap();
        assert!(e.add_part(id, &part).is_none(), "offset already taken");
        assert_ne!(pid, hit.part);
        assert!((e.mass(id).unwrap() - 3200.0).abs() < 1.0);
    }
}
