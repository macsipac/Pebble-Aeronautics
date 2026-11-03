//! Pumpkin glue: /physics commands (assemble, delete), block-display visuals, collision with entities,
//! breaking / placing blocks on physics objects.
use std::cell::{Cell as StdCell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use pumpkin_plugin_api::{
    BlockDisplayEntity, Context, DisplayTransformation, Entity, EntityDisplayExt, EntityType,
    ItemStack, ItemStackExt, PersistentDataHolder, Plugin, PluginMetadata, Quaternionf, Server, Vector3f,
    register_plugin,
    command::{CommandError, CommandSender, ConsumedArgs},
    command_wit::CommandSenderType,
    commands::{Command, CommandHandler},
    common::{BlockPos, GameMode, Hand},
    events::{EventHandler, EventPriority},
    events_wit::{EntityInteractionAction, InteractAction, PlayerAnimationEventData, PlayerInteractEntityEventData, PlayerInteractEventData, PlayerInteractUnknownEntityEventData, PlayerToggleSneakEventData},
    permission::{Permission, PermissionDefault, PermissionLevel},
    scheduler::SchedulerExt,
    text::TextComponent,
    world::{self, BlockFlags, World},
};
use pumpkin_plugin_api::events::player::player_interact::PlayerInteractEvent;
use pumpkin_plugin_api::events::player::player_interact_entity::PlayerInteractEntityEvent;
use pumpkin_plugin_api::events::player::player_interact_unknown_entity::PlayerInteractUnknownEntityEvent;
use pumpkin_plugin_api::java_packets::{CRemoveEntities, CSetEntityMetadata, CSpawnEntity, ClientboundPacket};
use pumpkin_plugin_api::data_components::DataComponent;
use pumpkin_plugin_api::uuid::Uuid;
use pumpkin_plugin_api::gui::Gui;
use pumpkin_plugin_api::Screen;
use pumpkin_plugin_api::inventory_wit::Inventory;
use pumpkin_plugin_api::wit::pumpkin::plugin::sounds::SoundCategory;
use pumpkin_plugin_api::events::player::player_animation::PlayerAnimationEvent;
use pumpkin_plugin_api::events::player::player_toggle_sneak::PlayerToggleSneakEvent;

use crate::engine::*;
use crate::materials::{friction_from_slipperiness, material_for};

/// Everyone may use `/physics assemble`; only ops (and command blocks / console) the rest.
const PERM_USE: &str = "physics:use";
const PERM_ADMIN: &str = "physics:admin";
const MAX_ASSEMBLE: usize = 100;
const MAX_PARTS: usize = 400;
const REACH: f32 = 6.5;
/// Survival/creative melee reach used when punching a physics block.
const BREAK_REACH: f32 = 5.0;
/// Players closer than this (blocks) get the fake collision shulkers.
const COLLIDER_VIEW: f64 = 48.0;
/// Protocol id of the shulker entity type.
const SHULKER_TYPE_ID: i32 = 115;
/// Dragging: holding right-click with an empty hand pulls the block towards the crosshair.
const GRAB_TIMEOUT_TICKS: u64 = 8;
const GRAB_GAIN: f32 = 4.0;
const GRAB_MAX_SPEED: f32 = 8.0;
const GRAB_BLEND: f32 = 0.15;
/// Scale of every physics block display. Marginally above 1 so neighbouring blocks overlap a hair
/// instead of showing seams / z-fighting.
const DISPLAY_SCALE: f32 = 1.0001;
/// A block breaks after this many punches within `PUNCH_WINDOW_TICKS`.
const PUNCHES_TO_BREAK: usize = 3;
const PUNCH_WINDOW_TICKS: u64 = 20;
/// Namespace/keys used to tag the entities this plugin spawns (so stale ones can be found again).
const TAG_NS: &str = "physics";
const TAG_SESSION: &str = "session";
/// How often (ticks) changed body poses are re-written into the display entities' saved data.
const REFRESH_EVERY_TICKS: u64 = 20;
/// How often (ticks) we scan for physics entities left over from a previous run / an unloaded chunk.
const RECOVER_EVERY_TICKS: u64 = 40;
/// A saved body whose parts are not all loaded yet waits this many scans for the rest before being
/// revived with what is there.
const MAX_WAIT_SCANS: u32 = 8;

struct PartVis {
    id: u32,
    offset: [i32; 3],
    state: u16,
    display: Entity,
    bd: BlockDisplayEntity,
    /// Collision for full-cube blocks (see [`Collider`]).
    col: Option<Rc<Collider>>,
    /// Raw NBT of the block entity this block had (chest contents etc.). Inert while it is a physics block.
    nbt: Option<Vec<u8>>,
    /// Contents of a chest-like block (see `snapshot`): `[slot, count, key length, key...]` per stack.
    inv: RefCell<Vec<u8>>,
    /// Inventory of the GUI that was opened last; read back to keep `inv` current.
    store: RefCell<Option<Inventory>>,
    /// Consecutive recovery scans in which this part's display was missing although its chunk is loaded.
    strikes: StdCell<u8>,
}
struct BodyVis {
    parts: RefCell<Vec<PartVis>>,
    last_rot: StdCell<[f32; 4]>,
    /// Part count last written into the entities' saved data (0 = must be rewritten).
    n_saved: StdCell<usize>,
    /// Pose last written into the entities' saved data.
    pose_saved: StdCell<Option<([f64; 3], [f32; 4])>>,
}
impl BodyVis {
    fn new(parts: Vec<PartVis>, last_rot: [f32; 4]) -> Self {
        Self { parts: RefCell::new(parts), last_rot: StdCell::new(last_rot), n_saved: StdCell::new(0), pose_saved: StdCell::new(None) }
    }
    /// Makes the next save rewrite the part count and pose.
    fn invalidate(&self) {
        self.n_saved.set(0);
        self.pose_saved.set(None);
    }
}
struct State {
    engines: HashMap<String, Engine>,
    bodies: HashMap<(String, BodyId), Rc<BodyVis>>,
    /// Saved bodies waiting for their other parts to load: (world, session, body) -> scans waited.
    waiting: HashMap<(String, i64, i64), u32>,
    ticks: u64,
    last_action: HashMap<u32, u64>,
    /// Per-player punch tracking for breaking.
    punches: HashMap<u32, Punch>,
    /// Bodies currently being dragged, by player entity id.
    grabs: HashMap<u32, Grab>,
    /// Entity ids of players that are crouching (the plugin API has no query, only the toggle event).
    sneaking: HashSet<u32>,
}
#[derive(Clone)]
struct Grab {
    world: String,
    body: BodyId,
    part: u32,
    /// Grabbed point in the body's local frame.
    local: [f32; 3],
    /// Distance from the eyes at which the block is held.
    dist: f64,
    /// Last tick a right-click packet was seen (holding the button repeats it).
    last: u64,
}
#[derive(Default)]
struct Punch {
    target: Option<(String, BodyId, u32)>,
    ticks: Vec<u64>,
}
thread_local! {
    static STATE: RefCell<State> = RefCell::new(State {
        engines: HashMap::new(), bodies: HashMap::new(), waiting: HashMap::new(), ticks: 0, last_action: HashMap::new(),
        punches: HashMap::new(), grabs: HashMap::new(), sneaking: HashSet::new(),
    });
}

// ---------------------------------------------------------------------------
// world access
// ---------------------------------------------------------------------------

struct View<'a> { world: &'a World }
impl WorldView for View<'_> {
    fn state_at(&mut self, c: Cell) -> Option<u16> {
        self.world.get_chunk(c[0] >> 4, c[2] >> 4)?;
        Some(self.world.get_block_state_id(BlockPos { x: c[0], y: c[1], z: c[2] }))
    }
    fn describe(&mut self, id: u16) -> StateInfo {
        let Some(s) = world::get_block_state_by_id(id) else { return StateInfo::empty() };
        let slip = world::get_block_by_id(s.block_id).map_or(0.6, |b| b.slipperiness);
        let mat = material_for(&s.block_name);
        let bx = |v: &Vec<world::BoundingBox>| -> Vec<[f32; 6]> {
            v.iter().map(|b| [b.min.0 as f32, b.min.1 as f32, b.min.2 as f32, b.max.0 as f32, b.max.1 as f32, b.max.2 as f32]).collect()
        };
        let waterlogged = s.properties.iter().any(|(k, v)| k == "waterlogged" && v == "true");
        let level = s.properties.iter().find(|(k, _)| k == "level").and_then(|(_, v)| v.parse().ok()).unwrap_or(0u8);
        let fluid = if s.block_name.ends_with(":water") || waterlogged { Some(Fluid::Water) }
            else if s.block_name.ends_with(":lava") { Some(Fluid::Lava) } else { None };
        let (boxes, outline) = if s.is_liquid { (Vec::new(), Vec::new()) } else { (bx(&s.collision_shapes), bx(&s.outline_shapes)) };
        StateInfo {
            boxes, outline, fluid, fluid_level: level,
            friction: friction_from_slipperiness(slip), restitution: mat.restitution, density: mat.density,
        }
    }
    fn min_y(&self) -> i32 { self.world.get_min_y() }
}

fn part_for(info: &StateInfo, offset: [i32; 3]) -> Part {
    let boxes = if !info.boxes.is_empty() { info.boxes.clone() }
        else if !info.outline.is_empty() { info.outline.clone() }
        else { vec![[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] };
    Part { offset, boxes, density: info.density, friction: info.friction, restitution: info.restitution }
}

/// True only for blocks whose collision shape is exactly one full cube (stone, planks, ...).
/// Only those get a collision shulker; slabs, stairs, fences etc. do not.
fn is_full_block(info: &StateInfo) -> bool {
    const EPS: f32 = 1e-3;
    info.boxes.len() == 1 && {
        let b = info.boxes[0];
        b[0] < EPS && b[1] < EPS && b[2] < EPS && b[3] > 1.0 - EPS && b[4] > 1.0 - EPS && b[5] > 1.0 - EPS
    }
}

thread_local! {
    /// Kept outside `STATE` so tagging entities never touches a `RefCell` that is already borrowed.
    static SESSION: StdCell<i64> = const { StdCell::new(0) };
}
fn session() -> i64 { SESSION.with(StdCell::get) }

/// Marks an entity as belonging to this plugin load. The mark is saved with the entity, so after a
/// restart anything carrying an old session id is a leftover and gets removed.
fn tag(e: &Entity) { e.set_long(TAG_NS, TAG_SESSION, session()); }

// ---------------------------------------------------------------------------
// visuals
// ---------------------------------------------------------------------------

/// Collision for one full-cube part: a shulker that exists only on clients (never in the world, so it
/// can't wander off, take damage or show up in the world data). It is announced to clients under the
/// entity id of the part's *anchor*, an invisible, empty block display at the block's bottom that does
/// exist on the server. That has three effects: the server's own position updates for the anchor move
/// the shulker on the client (so no per-tick packets of ours), Pumpkin finds a real entity when a
/// client punches the shulker (no "invalid entity" kick), and nothing extra has to be spawned.
struct Collider {
    anchor: Entity,
    /// The anchor's entity id, which the client-side shulker borrows.
    eid: i32,
    /// Where the shulker's feet should be at spawn time.
    target: StdCell<(f64, f64, f64)>,
    /// Players that currently have it spawned: (player entity id, tick sent).
    viewers: RefCell<Vec<(u32, u64)>>,
}
thread_local! {
    /// Removed colliders whose shulker still has to be despawned for their viewers.
    static GRAVE: RefCell<Vec<Rc<Collider>>> = const { RefCell::new(Vec::new()) };
    /// Ids of recently removed colliders: a client may still click the shulker for a moment.
    static RECENT_DEAD: RefCell<VecDeque<i32>> = const { RefCell::new(VecDeque::new()) };
    /// Tick at which each player was first seen by `sync_colliders` (player entity id -> tick).
    static FIRST_SEEN: RefCell<HashMap<u32, u64>> = RefCell::new(HashMap::new());
}
/// Ticks a player must have been online before they get raw collider packets. A player that shows
/// up in the player list can still be finishing the join sequence (login / configuration / world
/// load); a spawn packet arriving then is a protocol violation and the client disconnects.
const JOIN_GRACE_TICKS: u64 = 60;
fn spawn_packets(c: &Collider, at: (f64, f64, f64)) -> [ClientboundPacket; 2] {
    let n = c.eid as u64;
    [
        ClientboundPacket::CSpawnEntity(CSpawnEntity {
            entity_id: c.eid,
            entity_uuid: Uuid { high: 0xF4CE_5EED_0000_0000 | n, low: n },
            r_type: SHULKER_TYPE_ID,
            position: at,
            velocity: "[0.0,0.0,0.0]".to_string(),
            pitch: 0, yaw: 0, head_yaw: 0, data: 0,
        }),
        // Entity flags (index 0, type byte): 0x20 = invisible.
        ClientboundPacket::CSetEntityMetadata(CSetEntityMetadata { entity_id: c.eid, metadata: vec![0, 0, 0x20, 0xFF] }),
    ]
}
fn remove_packet(c: &Collider) -> ClientboundPacket {
    ClientboundPacket::CRemoveEntities(CRemoveEntities { entity_ids: vec![c.eid] })
}
fn dist_sq(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)
}
fn is_sneaking(player: &pumpkin_plugin_api::Player) -> bool {
    let id = player.as_entity().get_id();
    STATE.with(|s| s.borrow().sneaking.contains(&id))
}
fn now_tick() -> u64 { STATE.with(|s| s.borrow().ticks) }

/// Keeps every nearby Java player's client in sync with the collision shulkers.
fn sync_colliders(server: &Server) {
    let grave: Vec<Rc<Collider>> = GRAVE.with(|g| std::mem::take(&mut *g.borrow_mut()));
    let bodies: Vec<(String, Rc<BodyVis>)> = STATE.with(|s| s.borrow().bodies.iter().map(|((w, _), v)| (w.clone(), v.clone())).collect());
    if grave.is_empty() && bodies.is_empty() { return; }
    let now = now_tick();

    let mut players = Vec::new();
    for pl in server.get_all_players() {
        let Some(jp) = pl.as_java() else { continue };
        let e = pl.as_entity();
        let pid = e.get_id() as u32;
        let first = FIRST_SEEN.with(|f| *f.borrow_mut().entry(pid).or_insert(now));
        if now.saturating_sub(first) < JOIN_GRACE_TICKS { continue; }
        players.push((pid, pl.get_world().get_name(), e.get_eye_position(), jp));
    }
    // Forget players that left, so a reconnect (possibly with a reused id) gets the grace again.
    let online: HashSet<u32> = server.get_all_players().iter().map(|pl| pl.as_entity().get_id() as u32).collect();
    FIRST_SEEN.with(|f| f.borrow_mut().retain(|k, _| online.contains(k)));

    for c in &grave {
        for (pid, _) in c.viewers.borrow().iter() {
            if let Some((_, _, _, jp)) = players.iter().find(|p| p.0 == *pid) { jp.send_packet(&remove_packet(c)); }
        }
    }
    if players.is_empty() { return; }

    let resend = now % 200 == 0; // clients forget entities on respawn; re-announce now and then
    let view2 = COLLIDER_VIEW * COLLIDER_VIEW;
    for (wname, vis) in &bodies {
        let Ok(parts) = vis.parts.try_borrow() else { continue };
        for c in parts.iter().filter_map(|p| p.col.as_ref()) {
            let tgt = c.target.get();
            let mut viewers = c.viewers.borrow_mut();
            if resend { viewers.clear(); }
            for (pid, pw, eye, jp) in &players {
                let near = pw == wname && dist_sq(*eye, tgt) < view2;
                let idx = viewers.iter().position(|v| v.0 == *pid);
                match (near, idx) {
                    (true, None) => {
                        for pk in spawn_packets(c, tgt) { jp.send_packet(&pk); }
                        viewers.push((*pid, now));
                    }
                    // The server's own spawn of the anchor can reach the client after ours and replace the
                    // shulker there; announce it again a few times.
                    (true, Some(i)) if matches!(now.saturating_sub(viewers[i].1), 3 | 10 | 40) => {
                        for pk in spawn_packets(c, tgt) { jp.send_packet(&pk); }
                    }
                    (false, Some(i)) => {
                        jp.send_packet(&remove_packet(c));
                        viewers.remove(i);
                    }
                    _ => {}
                }
            }
            // Players that left the server / world: forget them.
            viewers.retain(|v| players.iter().any(|p| p.0 == v.0 && &p.1 == wname));
        }
    }
}

fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let t = [2.0 * (y * v[2] - z * v[1]), 2.0 * (z * v[0] - x * v[2]), 2.0 * (x * v[1] - y * v[0])];
    [v[0] + w * t[0] + (y * t[2] - z * t[1]), v[1] + w * t[1] + (z * t[0] - x * t[2]), v[2] + w * t[2] + (x * t[1] - y * t[0])]
}

/// Sets the block display's transform: scaled by `DISPLAY_SCALE` and rotated by `q` about the block's centre.
fn apply_rotation(bd: &BlockDisplayEntity, q: [f32; 4]) {
    // The model's origin is the block corner, while the display entity sits at the block's centre, so
    // shift by -R*(s/2): the block then rotates about the entity position.
    let h = -0.5 * DISPLAY_SCALE;
    let t = rotate(q, [h, h, h]);
    bd.get_display().set_transformation(DisplayTransformation {
        translation: Vector3f { x: t[0], y: t[1], z: t[2] },
        scale: Vector3f { x: DISPLAY_SCALE, y: DISPLAY_SCALE, z: DISPLAY_SCALE },
        left_rotation: Quaternionf { x: q[0], y: q[1], z: q[2], w: q[3] },
        right_rotation: Quaternionf { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
    });
}

fn set_rotation(p: &PartVis, q: [f32; 4]) { apply_rotation(&p.bd, q); }

fn part_center(pos: [f64; 3], rot: [f32; 4], offset: [i32; 3]) -> (f64, f64, f64) {
    let r = rotate(rot, [offset[0] as f32, offset[1] as f32, offset[2] as f32]);
    (pos[0] + f64::from(r[0]), pos[1] + f64::from(r[1]), pos[2] + f64::from(r[2]))
}

fn place_part_entities(p: &PartVis, pos: [f64; 3], rot: [f32; 4], with_rotation: bool) {
    let c = part_center(pos, rot, p.offset);
    // The display sits at the block's centre.
    p.display.teleport(c, p.display.get_world());
    // The anchor (and with it the shulker, whose feet are at its origin) sits at the block's bottom.
    if let Some(col) = &p.col {
        let feet = (c.0, c.1 - 0.5, c.2);
        col.anchor.teleport(feet, col.anchor.get_world());
        col.target.set(feet);
    }
    if with_rotation {
        set_rotation(p, rot);
    }
}

fn spawn_part(world: &World, state: u16, offset: [i32; 3], id: u32, center: (f64, f64, f64), solid: bool, nbt: Option<Vec<u8>>) -> Result<PartVis, String> {
    let display = world.spawn_entity(EntityType::BlockDisplay, center);
    let bd = display.as_block_display().ok_or("Could not create a block display")?;
    bd.set_block_state_id(state);
    let d = bd.get_display();
    d.set_interpolation_duration(2);
    d.set_teleport_duration(2);
    d.set_interpolation_start(0);
    apply_rotation(&bd, [0.0, 0.0, 0.0, 1.0]);
    display.set_has_gravity(false);
    tag(&display);

    // Collision (full-cube blocks only; a shulker is a full cube, slabs/stairs would feel wrong): see `Collider`.
    let feet = (center.0, center.1 - 0.5, center.2);
    let col = solid.then(|| {
        let a = world.spawn_entity(EntityType::BlockDisplay, feet);
        if let Some(abd) = a.as_block_display() { abd.get_display().set_teleport_duration(2); }
        a.set_has_gravity(false);
        tag(&a);
        Rc::new(Collider { eid: a.get_id() as i32, anchor: a, target: StdCell::new(feet), viewers: RefCell::new(Vec::new()) })
    });
    Ok(PartVis { id, offset, state, display, bd, col, nbt, inv: RefCell::new(Vec::new()), store: RefCell::new(None), strikes: StdCell::new(0) })
}

fn kill_part(p: PartVis) {
    p.display.remove();
    if let Some(c) = p.col {
        c.anchor.remove();
        RECENT_DEAD.with(|r| { let mut r = r.borrow_mut(); r.push_back(c.eid); if r.len() > 256 { r.pop_front(); } });
        GRAVE.with(|g| g.borrow_mut().push(c));
    }
}

// ---------------------------------------------------------------------------
// per-tick update
// ---------------------------------------------------------------------------

/// Moves an entity out of the body's blocks along the shortest way (separating-axis test of the entity's
/// box against each unit cube, on the cube's three axes). No server-side collider entity is involved.
fn push_out(e: &Entity, feet: (f64, f64, f64), hw: f64, h: f64, pos: [f64; 3], rot: [f32; 4], offsets: &[[i32; 3]]) {
    let conj = [-rot[0], -rot[1], -rot[2], rot[3]];
    // The world axes seen from the body's frame: they give the entity box's extent along each body axis.
    let (ax, ay, az) = (rotate(conj, [1.0, 0.0, 0.0]), rotate(conj, [0.0, 1.0, 0.0]), rotate(conj, [0.0, 0.0, 1.0]));
    let hh = h * 0.5;
    let reach = |i: usize| hw * f64::from(ax[i].abs()) + hh * f64::from(ay[i].abs()) + hw * f64::from(az[i].abs());
    let r = [reach(0), reach(1), reach(2)];
    let c = (feet.0, feet.1 + hh, feet.2);
    let mut best: Option<([f32; 3], f64)> = None;
    for off in offsets {
        let pc = part_center(pos, rot, *off);
        let l = rotate(conj, [(c.0 - pc.0) as f32, (c.1 - pc.1) as f32, (c.2 - pc.2) as f32]);
        let mut axis = 0;
        let mut depth = f64::MAX;
        let mut inside = true;
        for i in 0..3 {
            let d = 0.5 + r[i] - f64::from(l[i].abs());
            if d <= 0.0 { inside = false; break; }
            if d < depth { depth = d; axis = i; }
        }
        if !inside || best.as_ref().is_some_and(|b| b.1 <= depth) { continue; }
        let mut v = [0.0f32; 3];
        v[axis] = if l[axis] >= 0.0 { depth as f32 } else { -(depth as f32) };
        best = Some((v, depth));
    }
    if let Some((v, depth)) = best {
        if depth > 1.5 { return; } // deep inside (spawned in it): leave it alone rather than launch it
        let w = rotate(rot, v);
        e.teleport((feet.0 + f64::from(w[0]), feet.1 + f64::from(w[1]), feet.2 + f64::from(w[2])), e.get_world());
    }
}

fn collect_pushers(name: &str) -> Vec<PusherInput> {
    // (bounds, an entity belonging to the body to query around)
    let bodies: Vec<([f64; 3], [f64; 3], [f64; 3], [f32; 4], Rc<BodyVis>)> = STATE.with(|s| {
        let st = s.borrow();
        let Some(e) = st.engines.get(name) else { return Vec::new() };
        e.ids().into_iter().filter_map(|id| {
            let (lo, hi) = e.body_bounds(id)?;
            let (pos, rot, _) = e.body_state(id)?;
            let vis = st.bodies.get(&(name.to_string(), id))?.clone();
            Some((lo, hi, pos, rot, vis))
        }).collect()
    });
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (lo, hi, pos, rot, vis) in bodies {
        let Ok(parts) = vis.parts.try_borrow() else { continue };
        let Some(first) = parts.first() else { continue };
        let diag = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt() + 3.0;
        let offsets: Vec<[i32; 3]> = parts.iter().map(|p| p.offset).collect();
        for e in first.display.get_nearby_entities(diag, diag, diag) {
            let living = e.is_living();
            if !living && !matches!(e.get_type(), EntityType::Item) { continue; }
            let p = e.get_position();
            let near = p.0 > lo[0] - 2.0 && p.0 < hi[0] + 2.0 && p.1 > lo[1] - 2.5 && p.1 < hi[1] + 2.0 && p.2 > lo[2] - 2.0 && p.2 < hi[2] + 2.0;
            if !near || !seen.insert(e.get_id()) { continue; }
            // Mobs and items can't walk through the blocks: push them out (players have the client-side shulkers).
            if !matches!(e.get_type(), EntityType::Player) {
                push_out(&e, p, f64::from(e.get_width()) * 0.5, f64::from(e.get_height()), pos, rot, &offsets);
            }
            if !living { continue; }
            out.push(PusherInput { key: u64::from(e.get_id()), feet: [p.0, p.1, p.2], half_width: e.get_width() * 0.5, height: e.get_height() });
        }
    }
    out
}

/// One item for a broken part; it carries the block entity data (chest contents etc.) when there is any.
fn part_item(name: &str, nbt: Option<&[u8]>) -> ItemStack {
    let stack = ItemStack::of(name, 1);
    if let (Some(nbt), Some(comp)) = (nbt, comp_by_name(BLOCK_ENTITY_DATA)) {
        // Component wire format: a VarInt type id (0) followed by the unnamed NBT compound.
        let mut value = vec![0u8];
        value.extend_from_slice(nbt);
        stack.set_component(comp, &value);
    }
    stack
}

/// Is there room for one more of this item (a free slot or a stack with space)?
fn can_give(player: &pumpkin_plugin_api::Player, key: &str) -> bool {
    let inv = player.get_inventory().as_inventory();
    (0..inv.get_size().min(36)).any(|slot| match inv.get_item(slot) {
        None => true,
        Some(s) => norm(&s.get_registry_key()) == norm(key) && s.get_count() < s.get_max_count(),
    })
}

/// Registry names come with or without the `minecraft:` prefix depending on where they were read.
fn norm(key: &str) -> &str { key.strip_prefix("minecraft:").unwrap_or(key) }

/// Puts one item into the main inventory: onto a matching stack if there is room, else into a free slot.
/// Items with block-entity data (chest contents) never stack.
fn give(player: &pumpkin_plugin_api::Player, name: &str, nbt: Option<&[u8]>) {
    let inv = player.get_inventory().as_inventory();
    let size = inv.get_size().min(36);
    if nbt.is_none() {
        for slot in 0..size {
            if let Some(s) = inv.get_item(slot) {
                if norm(&s.get_registry_key()) == norm(name) && s.get_count() < s.get_max_count() {
                    s.set_count(s.get_count() + 1);
                    inv.set_item(slot, Some(s));
                    return;
                }
            }
        }
    }
    if let Some(slot) = (0..size).find(|i| inv.get_item(*i).is_none()) {
        inv.set_item(slot, Some(part_item(name, nbt)));
    }
}

// ---------------------------------------------------------------------------
// persistence: the display entities themselves carry the data
// ---------------------------------------------------------------------------
//
// The server already saves entities with their chunks, so each block display stores which body it
// belongs to, how many parts that body has, its offset, its block state and the body's pose (with a
// timestamp) in its own saved data. Two things can then go wrong, and both are handled by `recover`:
//
//  * Restart / plugin reload: the entities come back carrying an old session id. Their bodies are
//    rebuilt from the saved data (once all parts of a body are loaded, or after a while regardless),
//    and only then are the stale entities removed.
//  * A chunk unloads while the plugin runs: the entities are saved and despawned, the body keeps
//    simulating, and when the chunk loads again fresh copies appear that we don't know. Those are
//    removed and the body's visuals are rebuilt at its current pose.

fn now_millis() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}

/// Writes the data that never changes for a part (until it is moved to another body).
fn write_static(p: &PartVis, body: BodyId) {
    tag(&p.display);
    p.display.set_long(TAG_NS, "body", body as i64);
    p.display.set_int_array(TAG_NS, "off", p.offset.to_vec());
    p.display.set_int(TAG_NS, "state", i32::from(p.state));
    if let Some(n) = &p.nbt {
        p.display.set_byte_array(TAG_NS, "nbt", n.iter().map(|b| *b as i8).collect());
    }
    let inv = p.inv.borrow();
    if !inv.is_empty() { p.display.set_byte_array(TAG_NS, "inv", inv.iter().map(|b| *b as i8).collect()); }
}

// Component table, generated by build.rs from the `data-component` enum of the Pumpkin API being
// compiled against: `ALL_COMPONENTS: &[(&str, DataComponent)]` (WIT kebab-case name, value).
// Saves store the *name*, never an index, so adding / removing / reordering components in a nightly
// neither breaks the build nor corrupts old saves.
include!(concat!(env!("OUT_DIR"), "/components_gen.rs"));

/// The component order of the first save format (index-based, `INV_MAGIC`). Frozen forever: it only
/// maps old indices to names, as plain strings, so it cannot break the build.
const LEGACY_COMPONENT_NAMES: &[&str] = &[
    "custom-data",
    "max-stack-size",
    "max-damage",
    "damage",
    "unbreakable",
    "use-effects",
    "custom-name",
    "minimum-attack-charge",
    "damage-type",
    "item-name",
    "item-model",
    "lore",
    "rarity",
    "enchantments",
    "can-place-on",
    "can-break",
    "attribute-modifiers",
    "custom-model-data",
    "tooltip-display",
    "repair-cost",
    "creative-slot-lock",
    "enchantment-glint-override",
    "intangible-projectile",
    "food",
    "consumable",
    "use-remainder",
    "use-cooldown",
    "damage-resistant",
    "tool",
    "weapon",
    "attack-range",
    "enchantable",
    "equippable",
    "repairable",
    "glider",
    "tooltip-style",
    "death-protection",
    "blocks-attacks",
    "piercing-weapon",
    "kinetic-weapon",
    "attack-animation",
    "interact-animation",
    "additional-trade-cost",
    "block-transformer",
    "villager-food",
    "stored-enchantments",
    "dye",
    "dyed-color",
    "map-id",
    "map-decorations",
    "map-post-processing",
    "charged-projectiles",
    "bundle-contents",
    "potion-contents",
    "potion-duration-scale",
    "suspicious-stew-effects",
    "writable-book-content",
    "written-book-content",
    "trim",
    "debug-stick-state",
    "entity-data",
    "bucket-entity-data",
    "block-entity-data",
    "instrument",
    "provides-trim-material",
    "ominous-bottle-amplifier",
    "jukebox-playable",
    "provides-banner-patterns",
    "recipes",
    "lodestone-tracker",
    "firework-explosion",
    "fireworks",
    "profile",
    "note-block-sound",
    "banner-patterns",
    "base-color",
    "pot-decorations",
    "container",
    "block-state",
    "bees",
    "sulfur-cube-content",
    "lock",
    "container-loot",
    "break-sound",
    "compostable",
    "cooking-fuel",
    "brewing-fuel",
    "mob-visibility",
    "villager-variant",
    "wolf-variant",
    "wolf-sound-variant",
    "wolf-collar",
    "fox-variant",
    "salmon-size",
    "parrot-variant",
    "tropical-fish-pattern",
    "tropical-fish-base-color",
    "tropical-fish-pattern-color",
    "mooshroom-variant",
    "rabbit-variant",
    "pig-variant",
    "pig-sound-variant",
    "cow-variant",
    "cow-sound-variant",
    "chicken-variant",
    "chicken-sound-variant",
    "zombie-nautilus-variant",
    "frog-variant",
    "horse-variant",
    "painting-variant",
    "llama-variant",
    "axolotl-variant",
    "cat-variant",
    "cat-sound-variant",
    "cat-collar",
    "sheep-color",
    "shulker-color",
    "provides-pottery-pattern",
    "sign-text-front",
    "sign-text-back",
    "waxed",
    "cushion-color",
];

fn comp_by_name(name: &str) -> Option<DataComponent> {
    ALL_COMPONENTS.iter().find(|(n, _)| *n == name).map(|(_, c)| *c)
}
fn comp_name(c: DataComponent) -> Option<&'static str> {
    ALL_COMPONENTS.iter().find(|(_, v)| *v as u32 == c as u32).map(|(n, _)| *n)
}
fn is_comp(c: DataComponent, name: &str) -> bool {
    comp_by_name(name).is_some_and(|x| x as u32 == c as u32)
}
/// Component that carries a placed block's block-entity NBT (chest contents etc.).
const BLOCK_ENTITY_DATA: &str = "block-entity-data";


/// First component format: components stored by *index* into the component list. Still readable.
const INV_MAGIC: u8 = 0xFF;
/// Current format: components stored by *name*. (The very first format started with a slot number,
/// which never gets near 254.)
const INV_MAGIC_V2: u8 = 0xFE;

/// One saved stack of a chest-like block: everything the item carries (enchantments, names, nested
/// containers, ...) lives in `components`, verbatim.
struct StoredStack {
    slot: u32,
    key: String,
    count: u32,
    components: Vec<(DataComponent, Vec<u8>)>,
}

/// Saved format: `INV_MAGIC_V2`, then per stack
/// `[slot, count, key length, key..., component count, (name length, name..., u32 LE length, bytes...)...]`.
fn encode_inv(inv: &Inventory) -> Vec<u8> {
    let mut out = Vec::new();
    for slot in 0..inv.get_size().min(255) {
        let Some(st) = inv.get_item(slot) else { continue };
        let key = st.get_registry_key();
        let count = st.get_count() as u32;
        if count == 0 || key.len() > 255 { continue; }
        if out.is_empty() { out.push(INV_MAGIC_V2); }
        out.push(slot as u8);
        out.push(count.min(255) as u8);
        out.push(key.len() as u8);
        out.extend_from_slice(key.as_bytes());
        // A component this build cannot name is dropped rather than saved under a wrong name.
        let comps: Vec<_> = st.get_components().into_iter()
            .filter_map(|c| comp_name(c.component).map(|n| (n, c.value)))
            .take(255)
            .collect();
        out.push(comps.len() as u8);
        for (name, value) in comps {
            out.push(name.len() as u8);
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&(value.len() as u32).to_le_bytes());
            out.extend_from_slice(&value);
        }
    }
    out
}

/// Bounds-checked reader: corrupt or truncated saves end the parse, they never panic.
struct Rd<'a> { b: &'a [u8], i: usize }
impl<'a> Rd<'a> {
    fn u8(&mut self) -> Option<u8> { let v = *self.b.get(self.i)?; self.i += 1; Some(v) }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.i.checked_add(n)?;
        let s = self.b.get(self.i..end)?;
        self.i = end;
        Some(s)
    }
    fn u32_le(&mut self) -> Option<u32> { let s = self.take(4)?; Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]])) }
}

fn parse_inv(bytes: &[u8]) -> Vec<StoredStack> {
    let mut out = Vec::new();
    let mut r = Rd { b: bytes, i: 0 };
    match bytes.first() {
        Some(&INV_MAGIC_V2) | Some(&INV_MAGIC) => {
            let by_name = bytes[0] == INV_MAGIC_V2;
            r.i = 1;
            // Stacks parsed so far are kept even if a later one is damaged. A stack whose component
            // list is damaged still comes back, just with the components that were read.
            'stacks: while let (Some(slot), Some(count), Some(len)) = (r.u8(), r.u8(), r.u8()) {
                let Some(key) = r.take(len as usize).and_then(|k| std::str::from_utf8(k).ok()).map(str::to_string) else { break };
                let mut components = Vec::new();
                let mut damaged = false;
                match r.u8() {
                    None => damaged = true,
                    Some(ncomp) => for _ in 0..ncomp {
                        let name = if by_name {
                            let Some(n) = r.u8().and_then(|l| r.take(l as usize)) else { damaged = true; break };
                            std::str::from_utf8(n).ok().map(str::to_string)
                        } else {
                            let Some(i) = r.u8() else { damaged = true; break };
                            LEGACY_COMPONENT_NAMES.get(i as usize).map(|n| (*n).to_string())
                        };
                        let Some(value) = r.u32_le().and_then(|l| r.take(l as usize)) else { damaged = true; break };
                        // Unknown to this build (removed / renamed upstream): skip just this component.
                        if let Some(c) = name.as_deref().and_then(comp_by_name) { components.push((c, value.to_vec())); }
                    },
                }
                out.push(StoredStack { slot: u32::from(slot), key, count: u32::from(count), components });
                if damaged { break 'stacks; }
            }
        }
        _ => {
            // Oldest saves: `[slot, count, key length, key...]` only, no components.
            while let (Some(slot), Some(count), Some(len)) = (r.u8(), r.u8(), r.u8()) {
                let Some(key) = r.take(len as usize) else { break };
                if let Ok(key) = std::str::from_utf8(key) {
                    out.push(StoredStack { slot: u32::from(slot), key: key.to_string(), count: u32::from(count), components: Vec::new() });
                }
            }
        }
    }
    out
}

/// Rebuilds a saved stack, components and all.
fn stored_item(st: &StoredStack) -> ItemStack {
    let stack = ItemStack::of(&st.key, st.count as _);
    for (c, v) in &st.components { stack.set_component(*c, v); }
    stack
}

/// Reads the open GUI's inventory back into `inv` (and the saved data) when it changed.
fn snapshot(p: &PartVis) {
    let bytes = {
        let store = p.store.borrow();
        let Some(inv) = store.as_ref() else { return };
        encode_inv(inv)
    };
    if bytes != *p.inv.borrow() {
        p.display.set_byte_array(TAG_NS, "inv", bytes.iter().map(|b| *b as i8).collect());
        *p.inv.borrow_mut() = bytes;
    }
}

/// Keeps the saved contents of every opened chest-like block current.
fn sync_storage() {
    let all: Vec<Rc<BodyVis>> = STATE.with(|s| s.borrow().bodies.values().cloned().collect());
    for vis in all {
        let Ok(parts) = vis.parts.try_borrow() else { continue };
        for p in parts.iter() { snapshot(p); }
    }
}

/// Writes the body id / part count and the pose into the display entities, but only what changed
/// since the last write (or everything with `force`).
fn save_body(id: BodyId, vis: &BodyVis, pose: Option<([f64; 3], [f32; 4])>, force: bool) {
    let Ok(parts) = vis.parts.try_borrow() else { return };
    let n = parts.len();
    if force || vis.n_saved.get() != n {
        for p in parts.iter() {
            p.display.set_long(TAG_NS, "body", id as i64);
            p.display.set_int(TAG_NS, "n", n as i32);
        }
        vis.n_saved.set(n);
    }
    if let Some(pose) = pose {
        if force || vis.pose_saved.get() != Some(pose) {
            let mut v: Vec<i64> = pose.0.iter().map(|x| x.to_bits() as i64).collect();
            v.extend(pose.1.iter().map(|x| i64::from(x.to_bits())));
            v.push(now_millis());
            for p in parts.iter() {
                p.display.set_long_array(TAG_NS, "pose", v.clone());
            }
            vis.pose_saved.set(Some(pose));
        }
    }
}

/// Saves every body whose pose or part count changed (every body with `force`).
fn refresh_tags(force: bool) {
    let work: Vec<(BodyId, Rc<BodyVis>, Option<([f64; 3], [f32; 4])>)> = STATE.with(|s| {
        let st = s.borrow();
        st.bodies.iter().map(|((wname, id), vis)| {
            let pose = st.engines.get(wname).and_then(|e| e.body_state(*id)).map(|(pos, rot, _)| (pos, rot));
            (*id, vis.clone(), pose)
        }).collect()
    });
    for (id, vis, pose) in work { save_body(id, &vis, pose, force); }
}

/// Everything a saved display remembers.
struct Saved {
    body: i64,
    off: [i32; 3],
    state: u16,
    /// `[x, y, z, qx, qy, qz, qw]` as raw bits, plus (newer saves) a timestamp.
    pose: Vec<i64>,
    nbt: Option<Vec<u8>>,
    inv: Option<Vec<u8>>,
    /// How many parts the body had (0 = unknown, from an older save).
    n: usize,
}
fn read_saved(e: &Entity) -> Option<Saved> {
    let body = e.get_long(TAG_NS, "body")?;
    let off = e.get_int_array(TAG_NS, "off")?;
    let state = e.get_int(TAG_NS, "state")?;
    let pose = e.get_long_array(TAG_NS, "pose")?;
    if off.len() != 3 || !(pose.len() == 7 || pose.len() == 8) { return None; }
    let nbt = e.get_byte_array(TAG_NS, "nbt").map(|v| v.iter().map(|b| *b as u8).collect::<Vec<u8>>());
    let n = e.get_int(TAG_NS, "n").map_or(0, |v| v.max(0) as usize);
    let inv = e.get_byte_array(TAG_NS, "inv").map(|v| v.iter().map(|b| *b as u8).collect::<Vec<u8>>());
    Some(Saved { body, off: [off[0], off[1], off[2]], state: state as u16, pose, nbt, inv, n })
}

/// Throws away a body's visuals and spawns fresh ones at the body's current pose.
fn rebuild_body(world: &World, wname: &str, id: BodyId, vis: &Rc<BodyVis>) {
    let Some((pos, rot, _)) = STATE.with(|s| s.borrow().engines.get(wname).and_then(|e| e.body_state(id))) else { return };
    let old: Vec<PartVis> = vis.parts.borrow_mut().drain(..).collect();
    let mut fresh = Vec::with_capacity(old.len());
    for p in old {
        let (pid, off, state, solid, nbt) = (p.id, p.offset, p.state, p.col.is_some(), p.nbt.clone());
        snapshot(&p);
        let (inv, store) = (p.inv.take(), p.store.take());
        kill_part(p);
        if let Ok(np) = spawn_part(world, state, off, pid, part_center(pos, rot, off), solid, nbt) {
            *np.inv.borrow_mut() = inv;
            *np.store.borrow_mut() = store;
            place_part_entities(&np, pos, rot, true);
            fresh.push(np);
        }
    }
    for p in &fresh { write_static(p, id); }
    vis.parts.borrow_mut().extend(fresh);
    vis.last_rot.set(rot);
    save_body(id, vis, Some((pos, rot)), true);
}

/// One scan: revives bodies saved by a previous run and repairs bodies whose entities were lost to a
/// chunk unload.
fn recover(server: &Server) {
    let session = session();

    // Entity ids of every helper entity we currently own.
    let mut live: HashSet<u32> = HashSet::new();
    STATE.with(|s| {
        for vis in s.borrow().bodies.values() {
            if let Ok(parts) = vis.parts.try_borrow() {
                for p in parts.iter() {
                    live.insert(p.display.get_id());
                    if let Some(c) = &p.col { live.insert(c.anchor.get_id()); }
                }
            }
        }
    });

    struct Found { entity: Entity, saved: Saved }
    let mut loaded: HashSet<u32> = HashSet::new();
    let mut orphaned: HashSet<(String, i64)> = HashSet::new();
    let mut groups: HashMap<(String, i64, i64), Vec<Found>> = HashMap::new();
    for world in server.get_all_worlds() {
        let wname = world.get_name();
        for e in world.get_entities() {
            let id = e.get_id();
            loaded.insert(id);
            if live.contains(&id) { continue; }
            let Some(sess) = e.get_long(TAG_NS, TAG_SESSION) else { continue };
            match read_saved(&e) {
                // Ours from this run but unknown now: a copy that came back with its chunk.
                Some(sv) if sess == session => { orphaned.insert((wname.clone(), sv.body)); e.remove(); }
                // From an earlier run: revive below; it is removed only once that worked.
                Some(sv) => groups.entry((wname.clone(), sess, sv.body)).or_default().push(Found { entity: e, saved: sv }),
                // Hitboxes and leftovers carry no data.
                None => { e.remove(); }
            }
        }
    }

    // Bodies of this run whose display is gone although its chunk is loaded: rebuild the visuals.
    let bodies: Vec<(String, BodyId, Rc<BodyVis>)> = STATE.with(|s| s.borrow().bodies.iter().map(|((w, id), v)| (w.clone(), *id, v.clone())).collect());
    for (wname, id, vis) in bodies {
        let Some(world) = server.get_world_by_name(&wname) else { continue };
        let Some((pos, rot, _)) = STATE.with(|s| s.borrow().engines.get(&wname).and_then(|e| e.body_state(id))) else { continue };
        let copy_seen = orphaned.contains(&(wname.clone(), id as i64));
        let mut need = false;
        if let Ok(parts) = vis.parts.try_borrow() {
            for p in parts.iter() {
                let c = part_center(pos, rot, p.offset);
                let chunk_loaded = world.get_chunk((c.0.floor() as i32) >> 4, (c.2.floor() as i32) >> 4).is_some();
                if chunk_loaded && !loaded.contains(&p.display.get_id()) {
                    p.strikes.set(p.strikes.get().saturating_add(1));
                    // A copy of this body just reappeared, or the display has been missing for two scans.
                    if copy_seen || p.strikes.get() >= 2 { need = true; }
                } else {
                    p.strikes.set(0);
                }
            }
        }
        if need { rebuild_body(&world, &wname, id, &vis); }
    }

    // Bodies saved by an earlier run.
    for (key, found) in groups {
        let Some(world) = server.get_world_by_name(&key.0) else { continue };
        let expected = found.iter().map(|f| f.saved.n).max().unwrap_or(0);
        if found.len() < expected {
            // Part of the body is in a chunk that isn't loaded yet; give it a while.
            let waited = STATE.with(|s| { let mut st = s.borrow_mut(); let w = st.waiting.entry(key.clone()).or_insert(0); *w += 1; *w });
            if waited < MAX_WAIT_SCANS { continue; }
        }
        STATE.with(|s| { s.borrow_mut().waiting.remove(&key); });

        // Parts can have been saved at different moments; the newest pose is the right one.
        let Some(newest) = found.iter().max_by_key(|f| f.saved.pose.get(7).copied().unwrap_or(0)) else { continue };
        let pose = &newest.saved.pose;
        let pos = [f64::from_bits(pose[0] as u64), f64::from_bits(pose[1] as u64), f64::from_bits(pose[2] as u64)];
        let rot = [f32::from_bits(pose[3] as u32), f32::from_bits(pose[4] as u32), f32::from_bits(pose[5] as u32), f32::from_bits(pose[6] as u32)];
        let blocks: Vec<(Cell, u16)> = found.iter().map(|f| (f.saved.off, f.saved.state)).collect();
        let nbts: Vec<Option<Vec<u8>>> = found.iter().map(|f| f.saved.nbt.clone()).collect();
        // Revived bodies were allowed before, so they are not held to the part limit. The old entities
        // are only removed once the new body exists; if that fails they stay and we retry next scan.
        if let Ok(bid) = spawn_body_rotated(&world, &blocks, [0, 0, 0], Some(pos), Some(rot), &nbts) {
            if let Some(vis) = STATE.with(|s| s.borrow().bodies.get(&(key.0.clone(), bid)).cloned()) {
                for (i, f) in found.iter().enumerate() {
                    let Some(inv) = &f.saved.inv else { continue };
                    if let Some(p) = vis.parts.borrow().iter().find(|p| p.id == i as u32) {
                        *p.inv.borrow_mut() = inv.clone();
                        p.display.set_byte_array(TAG_NS, "inv", inv.iter().map(|b| *b as i8).collect());
                    }
                }
            }
            for f in found { f.entity.remove(); }
        }
    }
}

fn housekeeping(server: &Server) {
    let now = now_tick();
    if now % RECOVER_EVERY_TICKS == 1 { recover(server); }
    if now % REFRESH_EVERY_TICKS == 0 { refresh_tags(false); sync_storage(); }
    if now % RECOVER_EVERY_TICKS == 2 {
        // Forget crouch state of players that left.
        let online: HashSet<u32> = server.get_all_players().iter().map(|p| p.as_entity().get_id()).collect();
        STATE.with(|s| s.borrow_mut().sneaking.retain(|id| online.contains(id)));
    }
}

fn tick(server: Server) {
    STATE.with(|s| s.borrow_mut().ticks += 1);
    housekeeping(&server);
    apply_grabs(&server);
    let names: Vec<String> = STATE.with(|s| s.borrow().engines.iter().filter(|(_, e)| !e.is_empty()).map(|(n, _)| n.clone()).collect());
    for name in names {
        let Some(world) = server.get_world_by_name(&name) else { continue };
        let pushers = collect_pushers(&name);
        let updates = STATE.with(|s| {
            let mut st = s.borrow_mut();
            st.engines.get_mut(&name).map(|e| e.tick(&mut View { world: &world }, &pushers))
        }).unwrap_or_default();

        for u in updates {
            let vis = STATE.with(|s| s.borrow().bodies.get(&(name.clone(), u.id)).cloned());
            let Some(vis) = vis else { continue };
            if u.removed {
                STATE.with(|s| { s.borrow_mut().bodies.remove(&(name.clone(), u.id)); });
                let parts: Vec<PartVis> = vis.parts.borrow_mut().drain(..).collect();
                for p in parts {
                    kill_part(p);
                }
                continue;
            }
            let old = vis.last_rot.get();
            let dot = old[0] * u.rot[0] + old[1] * u.rot[1] + old[2] * u.rot[2] + old[3] * u.rot[3];
            let rot_changed = 1.0 - dot.abs() > 1e-7;
            vis.last_rot.set(u.rot);
            if let Ok(parts) = vis.parts.try_borrow() {
                for p in parts.iter() {
                    place_part_entities(p, u.pos, u.rot, rot_changed);
                }
            }
        }
    }
    sync_colliders(&server);
}

// ---------------------------------------------------------------------------
// registering bodies
// ---------------------------------------------------------------------------

/// Spawns visuals + a body for a set of `(cell, state)` blocks. `anchor` is the cell of the offset-0 block.
fn spawn_body_rotated(world: &World, blocks: &[(Cell, u16)], anchor: Cell, spawn_center: Option<[f64; 3]>, rot: Option<[f32; 4]>, nbts: &[Option<Vec<u8>>]) -> Result<BodyId, String> {
    let wname = world.get_name();
    let mut view = View { world };
    let center = spawn_center.unwrap_or([f64::from(anchor[0]) + 0.5, f64::from(anchor[1]) + 0.5, f64::from(anchor[2]) + 0.5]);

    let mut specs = Vec::new();
    let mut solid = Vec::new();
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        let eng = st.engines.entry(wname.clone()).or_insert_with(|| Engine::new(Config::default()));
        for (cell, state) in blocks {
            let info = eng.info(&mut view, *state);
            solid.push(is_full_block(&info));
            specs.push(part_for(&info, [cell[0] - anchor[0], cell[1] - anchor[1], cell[2] - anchor[2]]));
        }
    });

    let rq = rot.unwrap_or([0.0, 0.0, 0.0, 1.0]);
    let mut vis_parts = Vec::new();
    for (i, (cell, state)) in blocks.iter().enumerate() {
        let off = [cell[0] - anchor[0], cell[1] - anchor[1], cell[2] - anchor[2]];
        let c = part_center(center, rq, off);
        match spawn_part(world, *state, off, i as u32, c, solid[i], nbts.get(i).cloned().flatten()) {
            Ok(p) => vis_parts.push(p),
            Err(e) => {
                for p in vis_parts { kill_part(p); }
                return Err(e);
            }
        }
    }

    // A revived body may be rotated: put the visuals in the right place straight away instead of
    // waiting for the first physics update.
    for p in &vis_parts { place_part_entities(p, center, rq, true); }

    let (id, vis) = STATE.with(|s| {
        let mut st = s.borrow_mut();
        let eng = st.engines.get_mut(&wname).expect("engine");
        let id = eng.spawn(&BodySpec { parts: specs }, center);
        if let Some(r) = rot { eng.set_rotation(id, r); }
        for p in &vis_parts {
            write_static(p, id);
        }
        let vis = Rc::new(BodyVis::new(vis_parts, rq));
        st.bodies.insert((wname, id), vis.clone());
        (id, vis)
    });
    save_body(id, &vis, Some((center, rq)), true);
    Ok(id)
}

fn total_parts() -> usize {
    STATE.with(|s| s.borrow().bodies.values().map(|b| b.parts.try_borrow().map_or(0, |p| p.len())).sum())
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

fn fail(msg: &str) -> CommandError { CommandError::CommandFailed(TextComponent::text(msg)) }

fn is_admin(sender: &CommandSender, server: &Server) -> bool {
    match sender.get_command_sender_type() {
        // Only ops can place command blocks, and the console is always trusted.
        CommandSenderType::Console | CommandSenderType::Rcon | CommandSenderType::CommandBlock(_) => true,
        _ => sender.has_permission(server, PERM_ADMIN),
    }
}

struct DeleteCmd;
impl CommandHandler for DeleteCmd {
    fn handle(&self, sender: CommandSender, server: Server, _a: ConsumedArgs) -> Result<i32, CommandError> {
        if !is_admin(&sender, &server) { return Err(CommandError::PermissionDenied); }
        let (bodies, _) = STATE.with(|s| {
            let mut st = s.borrow_mut();
            for e in st.engines.values_mut() { e.clear(); }
            let b: Vec<Rc<BodyVis>> = st.bodies.drain().map(|(_, v)| v).collect();
            (b, ())
        });
        let mut n = 0;
        for b in bodies {
            let parts: Vec<PartVis> = b.parts.borrow_mut().drain(..).collect();
            for p in parts { n += 1; kill_part(p); }
        }
        sender.send_message(TextComponent::text(&format!("Removed {n} physics block(s)")));
        Ok(n)
    }
}

struct AssembleCmd;
impl CommandHandler for AssembleCmd {
    fn handle(&self, sender: CommandSender, _server: Server, _args: ConsumedArgs) -> Result<i32, CommandError> {
        let player = sender.as_player().ok_or_else(|| fail("Only players can assemble"))?;
        let world = player.get_world();
        let (fx, fy, fz) = player.as_entity().get_position();
        let mut start = [fx.floor() as i32, (fy - 0.001).floor() as i32, fz.floor() as i32];

        let solid = |c: Cell| -> Option<(u16, bool)> {
            let st = world.get_block_state(BlockPos { x: c[0], y: c[1], z: c[2] });
            if st.is_air || st.is_liquid { return None; }
            Some((st.id, st.hardness < 0.0))
        };
        if solid(start).is_none() {
            start[1] += 1;
            if solid(start).is_none() { return Err(fail("You must be standing on a block")); }
        }

        // Flood fill over face-adjacent blocks.
        let mut seen: HashSet<Cell> = HashSet::new();
        let mut order: Vec<(Cell, u16)> = Vec::new();
        let mut queue = VecDeque::from([start]);
        seen.insert(start);
        while let Some(c) = queue.pop_front() {
            let Some((state, unbreakable)) = solid(c) else { continue };
            if unbreakable { return Err(fail("That structure contains unbreakable blocks")); }
            order.push((c, state));
            if order.len() > MAX_ASSEMBLE {
                return Err(fail("More than 100 connected blocks, nothing was assembled"));
            }
            for d in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] {
                let n = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
                if seen.insert(n) { queue.push_back(n); }
            }
        }
        if total_parts() + order.len() > MAX_PARTS {
            return Err(fail("Too many physics blocks in the world right now (max 400)"));
        }

        // Block entities (chests, furnaces, signs...) keep their NBT; it is stored on the part and handed
        // back when the block is broken or placed again. They do not function while physics blocks.
        let nbts: Vec<Option<Vec<u8>>> = order.iter().map(|(c, _)| world.get_block_entity_nbt(BlockPos { x: c[0], y: c[1], z: c[2] })).collect();
        spawn_body_rotated(&world, &order, start, None, None, &nbts).map_err(|e| fail(&e))?;
        // Remove without drops / container-spill: the contents now live in the physics block.
        let flags = BlockFlags::NOTIFY_LISTENERS | BlockFlags::NOTIFY_NEIGHBORS | BlockFlags::SKIP_DROPS | BlockFlags::SKIP_BLOCK_ENTITY_REPLACED_CALLBACK;
        for (c, _) in &order {
            world.set_block_by_name(BlockPos { x: c[0], y: c[1], z: c[2] }, "minecraft:air", flags);
        }
        sender.send_message(TextComponent::text(&format!("Assembled {} block(s)", order.len())));
        Ok(order.len() as i32)
    }
}

// ---------------------------------------------------------------------------
// punching out / placing blocks on physics objects
// ---------------------------------------------------------------------------

fn look_dir(e: &Entity) -> [f32; 3] {
    let (yaw, pitch) = ((e.get_yaw() as f64).to_radians(), (e.get_pitch() as f64).to_radians());
    [(-yaw.sin() * pitch.cos()) as f32, (-pitch.sin()) as f32, (yaw.cos() * pitch.cos()) as f32]
}

/// Attacks are *not* handled through entity events: modern clients send punches in their own packet.
/// Punching is detected from the arm swing instead (see [`AnimHandler`]).
///
/// There is no hitbox entity: right-clicks are matched to physics blocks by ray-casting (see
/// [`UseHandler`]), and clicks on a full-cube block's client-only shulker arrive as
/// [`EntityHandler`] events.

/// True when `id` is the anchor entity whose id one of our client-only shulkers borrows.
fn is_collider(id: i32) -> bool {
    STATE.with(|s| s.borrow().bodies.values().any(|v| {
        v.parts.try_borrow().is_ok_and(|ps| ps.iter().any(|p| p.col.as_ref().is_some_and(|c| c.eid == id)))
    }))
}

/// Our client-only shulkers use the entity id of a real anchor, so Pumpkin finds the target of a punch
/// and doesn't kick for "attacking an invalid entity". Punches themselves are counted from the arm
/// swing; a right-click on the shulker is treated as a click on the block.
struct EntityHandler;
impl EventHandler<PlayerInteractEntityEvent> for EntityHandler {
    fn handle(&self, _server: Server, mut ev: PlayerInteractEntityEventData) -> PlayerInteractEntityEventData {
        if is_collider(ev.entity_id) {
            ev.cancelled = true;
            if !matches!(ev.action, EntityInteractionAction::Attack) {
                use_block(&ev.player);
            }
        }
        ev
    }
}

/// Safety net: a click on a shulker whose anchor was removed a moment ago would get the player kicked
/// for an unknown entity; swallow it instead.
struct UnknownEntityHandler;
impl EventHandler<PlayerInteractUnknownEntityEvent> for UnknownEntityHandler {
    fn handle(&self, _server: Server, mut ev: PlayerInteractUnknownEntityEventData) -> PlayerInteractUnknownEntityEventData {
        if RECENT_DEAD.with(|r| r.borrow().contains(&ev.entity_id)) {
            ev.cancelled = true;
        }
        ev
    }
}

/// Closest physics block under the crosshair, unless a real block is in front of it.
fn physics_hit(player: &pumpkin_plugin_api::Player, reach: f32) -> Option<Hit> {
    let world = player.get_world();
    let wname = world.get_name();
    let e = player.as_entity();
    let eye = e.get_eye_position();
    let dir = look_dir(&e);
    let hit = STATE.with(|s| s.try_borrow().ok().and_then(|st| st.engines.get(&wname).and_then(|eng| if eng.is_empty() { None } else { eng.raycast([eye.0, eye.1, eye.2], dir, reach) })))?;
    if shielded(&world, eye, dir, reach, hit.toi) { None } else { Some(hit) }
}

/// True when a real block lies between the eyes and a physics hit `toi` away.
fn shielded(world: &World, eye: (f64, f64, f64), dir: [f32; 3], reach: f32, toi: f32) -> bool {
    let r = f64::from(reach);
    let end = (eye.0 + f64::from(dir[0]) * r, eye.1 + f64::from(dir[1]) * r, eye.2 + f64::from(dir[2]) * r);
    world.ray_trace_blocks(eye, end).is_some_and(|b| dist_sq(b, eye).sqrt() + 0.05 < f64::from(toi))
}

/// Right-click (also repeated while the button is held, also in the air). Over a physics block it
/// places the held block onto it, or - with an empty hand - grabs it / keeps the drag alive. Every
/// event also keeps an existing drag alive, even when the crosshair has left the block.
struct UseHandler;
impl EventHandler<PlayerInteractEvent> for UseHandler {
    fn handle(&self, _server: Server, mut ev: PlayerInteractEventData) -> PlayerInteractEventData {
        refresh_grab(&ev.player);
        let right = matches!(ev.action, InteractAction::RightClickAir | InteractAction::RightClickBlock);
        if physics_hit(&ev.player, REACH).is_some() {
            // Left-clicks only punch (see AnimHandler); right-clicks open / place / drag. Whatever we handle
            // must not also act on the real block behind the physics block.
            ev.cancelled = if right { use_block(&ev.player) } else { true };
        }
        ev
    }
}

/// (place, break) sound identifiers for a block, picked from its name.
/// NOTE: these go through `play_custom_sound` (plain resource-location strings) because the host's
/// `play_sound(Sound)` builds the lookup name from the enum's Debug output ("blockwoodplace"),
/// never matches a real sound, and traps the whole guest ("Unknown sound").
fn block_sounds(name: &str) -> (&'static str, &'static str) {
    let n = norm(name);
    let any = |needles: &[&str]| needles.iter().any(|k| n.contains(k));
    if any(&["glass", "ice"]) { ("minecraft:block.glass.place", "minecraft:block.glass.break") }
    else if any(&["wool", "carpet"]) { ("minecraft:block.wool.place", "minecraft:block.wool.break") }
    else if any(&["snow"]) { ("minecraft:block.snow.place", "minecraft:block.snow.break") }
    else if n.contains("sand") && !n.contains("sandstone") { ("minecraft:block.sand.place", "minecraft:block.sand.break") }
    else if any(&["gravel", "dirt", "clay", "concrete_powder"]) { ("minecraft:block.gravel.place", "minecraft:block.gravel.break") }
    else if any(&["grass", "leaves", "moss", "hay_block"]) { ("minecraft:block.grass.place", "minecraft:block.grass.break") }
    else if any(&["slime"]) { ("minecraft:block.slime_block.place", "minecraft:block.slime_block.break") }
    else if any(&["honey"]) { ("minecraft:block.honey_block.place", "minecraft:block.honey_block.break") }
    else if any(&["bone_block"]) { ("minecraft:block.bone_block.place", "minecraft:block.bone_block.break") }
    else if any(&["netherrack"]) { ("minecraft:block.netherrack.place", "minecraft:block.netherrack.break") }
    else if any(&["deepslate"]) { ("minecraft:block.deepslate.place", "minecraft:block.deepslate.break") }
    else if any(&["copper"]) { ("minecraft:block.copper.place", "minecraft:block.copper.break") }
    else if any(&["amethyst"]) { ("minecraft:block.amethyst_block.place", "minecraft:block.amethyst_block.break") }
    else if n == "mud" || n.contains("mud_") { ("minecraft:block.mud.place", "minecraft:block.mud.break") }
    else if any(&["iron", "gold_block", "diamond_block", "emerald_block", "lapis_block", "netherite", "anvil", "chain", "redstone_block"]) { ("minecraft:block.metal.place", "minecraft:block.metal.break") }
    else if any(&["planks", "log", "wood", "stem", "hyphae", "door", "fence", "chest", "barrel", "crafting_table", "bookshelf", "jukebox", "note_block", "bamboo", "lectern", "loom", "composter"]) { ("minecraft:block.wood.place", "minecraft:block.wood.break") }
    else { ("minecraft:block.stone.place", "minecraft:block.stone.break") }
}

/// What right-clicking a block does (when not crouching).
enum Interaction { Storage(Screen, &'static str), EnderChest }
fn interaction(name: &str) -> Option<Interaction> {
    let n = norm(name);
    match n {
        "chest" | "trapped_chest" => Some(Interaction::Storage(Screen::Generic9x3, "Chest")),
        "barrel" => Some(Interaction::Storage(Screen::Generic9x3, "Barrel")),
        "ender_chest" => Some(Interaction::EnderChest),
        _ if n.ends_with("shulker_box") => Some(Interaction::Storage(Screen::ShulkerBox, "Shulker Box")),
        _ => None,
    }
}

/// Opens the block under the crosshair if it is something you can open. True when it did.
fn try_interact(player: &pumpkin_plugin_api::Player, wname: &str, now: u64) -> bool {
    let e = player.as_entity();
    let (eye, dir) = (e.get_eye_position(), look_dir(&e));
    let Some(hit) = STATE.with(|s| s.borrow().engines.get(wname).and_then(|eng| eng.raycast([eye.0, eye.1, eye.2], dir, REACH))) else { return false };
    let Some(vis) = STATE.with(|s| s.borrow().bodies.get(&(wname.to_string(), hit.body)).cloned()) else { return false };
    let parts = vis.parts.borrow();
    let Some(p) = parts.iter().find(|p| p.id == hit.part) else { return false };
    let Some(kind) = world::block_state_to_info(p.state).and_then(|i| interaction(&i.name)) else { return false };

    let pid = e.get_id();
    if STATE.with(|s| s.borrow().last_action.get(&pid).is_some_and(|t| now.saturating_sub(*t) < 8)) { return true; }
    STATE.with(|s| { s.borrow_mut().last_action.insert(pid, now); });
    match kind {
        Interaction::EnderChest => player.open_ender_chest(),
        Interaction::Storage(screen, title) => {
            snapshot(p); // pick up what the last viewer left in it
            let gui = Gui::new(screen, TextComponent::text(title));
            let inv = gui.get_inventory();
            let size = inv.get_size();
            for st in parse_inv(&p.inv.borrow()) {
                if st.slot < size { gui.set_item(st.slot, stored_item(&st)); }
            }
            *p.store.borrow_mut() = Some(inv);
            player.open_gui(gui);
        }
    }
    true
}

/// A right-click on a physics block. Crouching skips opening it (so a block can be placed against a chest
/// and the block can be dragged). True when the click did something.
fn use_block(player: &pumpkin_plugin_api::Player) -> bool {
    let world = player.get_world();
    let wname = world.get_name();
    let pid = player.as_entity().get_id();
    let now = now_tick();

    let sneaking = is_sneaking(player);
    if !sneaking && try_interact(player, &wname, now) { return true; }
    if player.get_item_in_hand(Hand::Right).is_none() {
        // Dragging needs crouching; an uncrouched empty-handed click is left alone.
        if sneaking { start_grab(player, &wname, now); }
        return sneaking;
    }
    let recent = STATE.with(|s| s.borrow().last_action.get(&pid).is_some_and(|t| now.saturating_sub(*t) < 3));
    if recent { return true; }
    STATE.with(|s| { s.borrow_mut().last_action.insert(pid, now); });
    place_part(&world, &wname, 0, player);
    true
}

fn start_grab(player: &pumpkin_plugin_api::Player, wname: &str, now: u64) {
    let e = player.as_entity();
    let pid = e.get_id();
    if STATE.with(|s| { let mut st = s.borrow_mut(); if let Some(g) = st.grabs.get_mut(&pid) { g.last = now; true } else { false } }) {
        return;
    }
    let eye = e.get_eye_position();
    let dir = look_dir(&e);
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        let Some(eng) = st.engines.get(wname) else { return };
        let Some(hit) = eng.raycast([eye.0, eye.1, eye.2], dir, REACH) else { return };
        let Some(local) = eng.grab_point_local(hit.body, hit.point) else { return };
        st.grabs.insert(pid, Grab { world: wname.to_string(), body: hit.body, part: hit.part, local, dist: f64::from(hit.toi), last: now });
    });
}

fn refresh_grab(player: &pumpkin_plugin_api::Player) {
    let pid = player.as_entity().get_id();
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        let now = st.ticks;
        if let Some(g) = st.grabs.get_mut(&pid) { g.last = now; }
    });
}

/// Pulls every grabbed body towards where its player is looking.
fn apply_grabs(server: &Server) {
    if STATE.with(|s| s.borrow().grabs.is_empty()) { return; }
    let players = server.get_all_players();
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        let now = st.ticks;
        let ids: Vec<u32> = st.grabs.keys().copied().collect();
        for pid in ids {
            let g = st.grabs[&pid].clone();
            let pl = players.iter().find(|p| p.as_entity().get_id() == pid);
            let alive = pl.filter(|p| {
                now.saturating_sub(g.last) <= GRAB_TIMEOUT_TICKS
                    && p.get_world().get_name() == g.world
                    && p.get_item_in_hand(Hand::Right).is_none()
                    && st.sneaking.contains(&pid)
            });
            let ok = alive.is_some_and(|p| {
                let e = p.as_entity();
                let eye = e.get_eye_position();
                let d = look_dir(&e);
                let target = [eye.0 + f64::from(d[0]) * g.dist, eye.1 + f64::from(d[1]) * g.dist, eye.2 + f64::from(d[2]) * g.dist];
                st.engines.get_mut(&g.world).is_some_and(|eng| eng.drag(g.body, g.local, target, GRAB_GAIN, GRAB_MAX_SPEED, GRAB_BLEND))
            });
            if !ok {
                st.grabs.remove(&pid);
            }
        }
    });
}

/// Remembers who is crouching.
struct SneakHandler;
impl EventHandler<PlayerToggleSneakEvent> for SneakHandler {
    fn handle(&self, _server: Server, ev: PlayerToggleSneakEventData) -> PlayerToggleSneakEventData {
        let id = ev.player.as_entity().get_id();
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            if ev.is_sneaking { st.sneaking.insert(id); } else { st.sneaking.remove(&id); }
        });
        ev
    }
}

/// Every left-click swings the arm, whether it hits a block, an entity or nothing. We ray-cast
/// against the physics bodies ourselves; hitting the same block `PUNCHES_TO_BREAK` times within one
/// second breaks it.
struct AnimHandler;
impl EventHandler<PlayerAnimationEvent> for AnimHandler {
    fn handle(&self, _server: Server, ev: PlayerAnimationEventData) -> PlayerAnimationEventData {
        if ev.animation_type == "ArmSwingMain" {
            punch(&ev.player);
        }
        ev
    }
}

fn punch(player: &pumpkin_plugin_api::Player) {
    let world = player.get_world();
    let wname = world.get_name();
    // Cheap early-out for the (very common) case of swinging with no physics blocks around.
    let any = STATE.with(|s| s.try_borrow().is_ok_and(|st| st.engines.get(&wname).is_some_and(|e| !e.is_empty())));
    if !any { return; }

    let e = player.as_entity();
    let eye = e.get_eye_position();
    let dir = look_dir(&e);
    let hit = STATE.with(|s| s.borrow().engines.get(&wname).and_then(|eng| eng.raycast([eye.0, eye.1, eye.2], dir, BREAK_REACH)));
    let Some(hit) = hit else {
        STATE.with(|s| { s.borrow_mut().punches.remove(&e.get_id()); });
        return;
    };
    if shielded(&world, eye, dir, BREAK_REACH, hit.toi) { return; }

    let pid = e.get_id();
    let key = (wname.clone(), hit.body, hit.part);
    let count = STATE.with(|s| {
        let mut st = s.borrow_mut();
        let now = st.ticks;
        let p = st.punches.entry(pid).or_default();
        if p.target.as_ref() != Some(&key) {
            p.target = Some(key.clone());
            p.ticks.clear();
        }
        p.ticks.retain(|t| now.saturating_sub(*t) < PUNCH_WINDOW_TICKS);
        p.ticks.push(now);
        p.ticks.len()
    });
    if count >= PUNCHES_TO_BREAK {
        STATE.with(|s| { s.borrow_mut().punches.remove(&pid); });
        break_part(&wname, hit.body, hit.part, player);
    }
}

fn break_part(wname: &str, body: BodyId, part: u32, player: &pumpkin_plugin_api::Player) {
    let Some(vis) = STATE.with(|s| s.borrow().bodies.get(&(wname.to_string(), body)).cloned()) else { return };
    let Some((state, nbt, offset)) = vis.parts.borrow().iter().find(|p| p.id == part).map(|p| (p.state, p.nbt.clone(), p.offset)) else { return };
    // A chest-like block with things inside can't be broken (the items would be lost).
    if vis.parts.borrow().iter().find(|p| p.id == part).is_some_and(|p| { snapshot(p); !p.inv.borrow().is_empty() }) {
        player.send_system_message(TextComponent::text("Empty it first"), true);
        return;
    }
    let at = STATE.with(|s| s.borrow().engines.get(wname).and_then(|e| e.body_state(body))).map(|(pos, rot, _)| part_center(pos, rot, offset));

    // Survival: the item goes straight into the inventory, so refuse to break when there is no room
    // instead of destroying the block. A block with block-entity data needs a free slot (it never stacks).
    let item = if player.get_gamemode() == GameMode::Creative { None } else { world::block_state_to_info(state).map(|i| i.name) };
    if let Some(name) = &item {
        let room = if nbt.is_some() { (0..player.get_inventory().as_inventory().get_size().min(36)).any(|i| player.get_inventory().as_inventory().get_item(i).is_none()) } else { can_give(player, name) };
        if !room {
            player.send_system_message(TextComponent::text("Your inventory is full"), true);
            return;
        }
    }

    let result = STATE.with(|s| s.borrow_mut().engines.get_mut(wname).map(|e| e.remove_part(body, part)));
    let Some(result) = result else { return };
    if matches!(result, Removed::NotFound) { return; }

    // Sound first: nothing below (inventory, visuals, saved data) may swallow it.
    if let (Some(at), Some(info)) = (at, world::block_state_to_info(state)) {
        player.get_world().play_custom_sound(block_sounds(&info.name).1, SoundCategory::Blocks, at, 1.0, 0.8);
    }

    // Remove the visual part (display and collision).
    let removed = {
        let mut parts = vis.parts.borrow_mut();
        parts.iter().position(|p| p.id == part).map(|i| parts.remove(i))
    };
    if let Some(p) = removed {
        kill_part(p);
    }
    vis.invalidate(); // part count changed: rewrite it into the saved data

    if let Some(name) = item { give(player, &name, nbt.as_deref()); }
    match result {
        Removed::BodyGone => {
            STATE.with(|s| { s.borrow_mut().bodies.remove(&(wname.to_string(), body)); });
        }
        Removed::Ok(splits) => {
            for sp in splits {
                let moved: Vec<PartVis> = {
                    let mut parts = vis.parts.borrow_mut();
                    let mut out = Vec::new();
                    let mut i = 0;
                    while i < parts.len() {
                        if sp.parts.contains(&parts[i].id) { out.push(parts.remove(i)); } else { i += 1; }
                    }
                    out
                };
                STATE.with(|s| {
                    let mut st = s.borrow_mut();
                    for p in &moved {
                        write_static(p, sp.body);
                    }
                    st.bodies.insert((wname.to_string(), sp.body), Rc::new(BodyVis::new(moved, vis.last_rot.get())));
                });
            }
        }
        Removed::NotFound => {}
    }
}

fn place_part(world: &World, wname: &str, _clicked_body: BodyId, player: &pumpkin_plugin_api::Player) {
    // What is the player holding, and is it a block?
    let Some(stack) = player.get_item_in_hand(Hand::Right) else { return };
    let Some(block) = world::get_block_by_name(&stack.get_registry_key()) else { return };
    let state = block.default_state_id;
    // An item that carries block entity data (e.g. a chest broken out of a physics body) keeps it.
    let held_nbt: Option<Vec<u8>> = stack
        .get_components()
        .into_iter()
        .find(|c| is_comp(c.component, BLOCK_ENTITY_DATA))
        .and_then(|c| c.value.split_first().map(|(_, rest)| rest.to_vec()))
        .filter(|v| !v.is_empty());

    // Which face of which block are they looking at?
    let e = player.as_entity();
    let (ex, ey, ez) = e.get_eye_position();
    let dir = look_dir(&e);
    let hit = STATE.with(|s| s.borrow().engines.get(wname).and_then(|eng| eng.raycast([ex, ey, ez], dir, REACH)));
    let Some(hit) = hit else { return };

    let Some(vis) = STATE.with(|s| s.borrow().bodies.get(&(wname.to_string(), hit.body)).cloned()) else { return };
    let Some(base) = vis.parts.borrow().iter().find(|p| p.id == hit.part).map(|p| p.offset) else { return };
    let off = [base[0] + hit.normal_local[0], base[1] + hit.normal_local[1], base[2] + hit.normal_local[2]];
    if total_parts() + 1 > MAX_PARTS { return; }

    let mut view = View { world };
    let part = STATE.with(|s| {
        let mut st = s.borrow_mut();
        let eng = st.engines.get_mut(wname)?;
        let info = eng.info(&mut view, state);
        let part = part_for(&info, off);
        let pid = eng.add_part(hit.body, &part)?;
        let bs = eng.body_state(hit.body)?;
        Some((pid, bs, is_full_block(&info)))
    });
    let Some((pid, (pos, rot, _), solid)) = part else { return };

    let center = part_center(pos, rot, off);
    match spawn_part(world, state, off, pid, center, solid, held_nbt) {
        Ok(p) => {
            place_part_entities(&p, pos, rot, true);
            write_static(&p, hit.body);
            vis.parts.borrow_mut().push(p);
            vis.invalidate(); // new part: it needs the part count and pose in its saved data
            if let Some(info) = world::block_state_to_info(state) {
                world.play_custom_sound(block_sounds(&info.name).0, SoundCategory::Blocks, center, 1.0, 0.8);
            }
        }
        Err(_) => {
            STATE.with(|s| { if let Some(e) = s.borrow_mut().engines.get_mut(wname) { let _ = e.remove_part(hit.body, pid); } });
            return;
        }
    }

    // Use up the item in survival.
    if player.get_gamemode() != GameMode::Creative {
        let n = stack.get_count();
        if n > 1 {
            stack.set_count(n - 1);
            player.set_item_in_hand(Hand::Right, Some(stack));
        } else {
            player.set_item_in_hand(Hand::Right, None);
        }
    }
}

// ---------------------------------------------------------------------------
// plugin
// ---------------------------------------------------------------------------

struct Physics;
impl Plugin for Physics {
    fn new() -> Self { Physics }
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "physics".into(), version: "0.3.0".into(), authors: vec![],
            description: "Rigid-body physics for block displays (/physics)".into(),
            dependencies: vec![], permissions: vec![],
        }
    }
    fn on_load(&self, ctx: Context) -> Result<(), String> {
        let session = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_millis() as i64).max(1);
        SESSION.with(|c| c.set(session));
        ctx.register_permission(&Permission {
            node: PERM_USE.into(), description: "Use /physics assemble".into(),
            default: PermissionDefault::Allow, children: vec![],
        })?;
        ctx.register_permission(&Permission {
            node: PERM_ADMIN.into(), description: "Use /physics delete".into(),
            default: PermissionDefault::Op(PermissionLevel::Two), children: vec![],
        })?;
        use pumpkin_plugin_api::command::CommandNode;
        let cmd = Command::new(&["physics".to_string()], "Blocks with real physics")
            .then(CommandNode::literal("delete").execute(DeleteCmd))
            .then(CommandNode::literal("assemble").execute(AssembleCmd));
        ctx.register_command(cmd, PERM_USE);
        ctx.register_event_handler::<PlayerToggleSneakEvent, _>(SneakHandler, EventPriority::Normal, false)?;
        ctx.register_event_handler::<PlayerAnimationEvent, _>(AnimHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<PlayerInteractEntityEvent, _>(EntityHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<PlayerInteractUnknownEntityEvent, _>(UnknownEntityHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<PlayerInteractEvent, _>(UseHandler, EventPriority::Normal, true)?;
        ctx.schedule_repeating_task(1, 1, tick);
        Ok(())
    }
    fn on_unload(&self, _ctx: Context) -> Result<(), String> {
        // Leave the entities in the world: they carry their own saved data and are revived by the
        // next load. Just make sure that data is current.
        refresh_tags(true);
        STATE.with(|s| {
            let mut st = s.borrow_mut();
            st.engines.clear();
            st.bodies.clear();
        });
        Ok(())
    }
}
register_plugin!(Physics);
