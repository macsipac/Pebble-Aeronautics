//! Pumpkin glue: /physics commands (assemble, delete), block-display visuals, collision with entities,
//! breaking / placing blocks on physics objects.
use std::cell::{Cell as StdCell, RefCell};
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::time::Instant;

use pumpkin_plugin_api::{
    BlockDisplayEntity, Context, DisplayTransformation, Entity, EntityDisplayExt, EntityType,
    ItemStack, ItemStackExt, PersistentDataHolder, Plugin, PluginMetadata, Quaternionf, Server, Vector3f,
    register_plugin,
    command::{CommandError, CommandSender, ConsumedArgs},
    command_wit::CommandSenderType,
    commands::{Command, CommandHandler},
    common::{BlockPos, GameMode, Hand},
    events::{EventHandler, EventPriority},
    events_wit::{EntityExplodeEventData, EntityInteractionAction, InteractAction, PlayerAnimationEventData, PlayerInteractEntityEventData, PlayerInteractEventData, PlayerInteractUnknownEntityEventData, PlayerToggleSneakEventData},
    permission::{Permission, PermissionDefault, PermissionLevel},
    scheduler::SchedulerExt,
    text::TextComponent,
    world::{self, BlockFlags, World},
};
use pumpkin_plugin_api::events::entity::entity_explode::EntityExplodeEvent;
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
const REFRESH_EVERY_TICKS: u64 = 60;
/// How often (ticks) we scan for physics entities left over from a previous run / an unloaded chunk.
const RECOVER_EVERY_TICKS: u64 = 60;
/// The entity scan walks every entity of every world through the host, which is by far the most
/// expensive thing this plugin does outside the physics tick, so it only runs when it can find something:
/// - at most every `SCAN_MIN_GAP_TICKS` after a player's chunk changed (new chunks may have loaded),
/// - every `RECOVER_EVERY_TICKS` while a previous scan left work open (waiting parts, a display strike),
/// - every `SCAN_IDLE_TICKS` as a safety net (chunks loaded by something other than a player).
const SCAN_MIN_GAP_TICKS: u64 = 120;
const SCAN_IDLE_TICKS: u64 = 1200;
/// A saved body whose parts are not all loaded yet waits this many scans for the rest before being
/// revived with what is there.
const MAX_WAIT_SCANS: u32 = 6;
/// Visuals only move when a body moved more than this (blocks) since they were last placed.
const POS_EPS: f64 = 1e-3;


struct PartVis {
    id: u32,
    /// Entity id of `display` (cached: asking the host costs a round trip).
    did: u32,
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
    /// Position the display entities were last teleported for (NaN = never, so the first update applies).
    last_pos: StdCell<[f64; 3]>,
}
impl BodyVis {
    fn new(parts: Vec<PartVis>, last_rot: [f32; 4]) -> Self {
        Self { parts: RefCell::new(parts), last_rot: StdCell::new(last_rot), n_saved: StdCell::new(0), pose_saved: StdCell::new(None), last_pos: StdCell::new([f64::NAN; 3]) }
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
    /// Tick of the last entity scan, whether the last scan left work open, and whether a player changed
    /// chunk since (see `scan_due`).
    last_scan: u64,
    scan_again: bool,
    chunks_dirty: bool,
    player_chunks: Vec<(u32, i32, i32)>,
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
        engines: HashMap::new(), bodies: HashMap::new(), waiting: HashMap::new(), ticks: 0, last_scan: 0, scan_again: false, chunks_dirty: false, player_chunks: Vec::new(), last_action: HashMap::new(),
        punches: HashMap::new(), grabs: HashMap::new(), sneaking: HashSet::new(),
    });
}

// ---------------------------------------------------------------------------
// world access
// ---------------------------------------------------------------------------

/// World access for the engine. Every call into the host is expensive, so the answers are memoised for
/// the lifetime of the `View` (one engine tick): terrain is queried in tight clusters, and many cells
/// and chunks are asked for several times (several bodies, several substeps).
struct View<'a> {
    world: &'a World,
    chunks: HashMap<(i32, i32), bool>,
    states: HashMap<(i32, i32, i32), u16>,
}
impl<'a> View<'a> {
    fn new(world: &'a World) -> Self {
        Self { world, chunks: HashMap::new(), states: HashMap::new() }
    }
}
impl WorldView for View<'_> {
    fn state_at(&mut self, c: Cell) -> Option<u16> {
        let key = (c[0], c[1], c[2]);
        if let Some(s) = self.states.get(&key) { return Some(*s); }
        let ck = (c[0] >> 4, c[2] >> 4);
        let loaded = match self.chunks.get(&ck) {
            Some(l) => *l,
            None => {
                let l = self.world.get_chunk(ck.0, ck.1).is_some();
                self.chunks.insert(ck, l);
                l
            }
        };
        if !loaded { return None; }
        let st = self.world.get_block_state_id(BlockPos { x: c[0], y: c[1], z: c[2] });
        self.states.insert(key, st);
        Some(st)
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
    /// Entity ids of every display and collider anchor this plugin owns. Entity queries return them too,
    /// and telling "ours" from "somebody else's" by id costs one host call instead of several.
    static OWN_IDS: RefCell<HashSet<u32>> = RefCell::new(HashSet::new());
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
    static FIRST_SEEN: RefCell<HashMap<u32, JoinInfo>> = RefCell::new(HashMap::new());
}
/// Ticks a player must have been online before they get raw collider packets. A player that shows
/// up in the player list can still be finishing the join sequence (login / configuration / world
/// load); a spawn packet arriving then is a protocol violation and the client disconnects.
const JOIN_GRACE_TICKS: u64 = 200;
/// What `sync_colliders` knows about a player's join: when it first saw them in this world, where they
/// stood and looked on the previous tick (kept in memory only, refreshed every tick while the player is
/// not ready yet; any change from it means the client is in the world), and whether they are cleared
/// for packets.
struct JoinInfo {
    first: u64,
    world: String,
    pos: (f64, f64, f64),
    look: (f32, f32),
    ready: bool,
}
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

/// The collider proximity pass runs every this many ticks (spawn/despawn of client-side shulkers does
/// not need tick precision; the shulkers themselves are moved by the server's anchor teleports).
const COLLIDER_SYNC_EVERY: u64 = 2;

/// Keeps every nearby Java player's client in sync with the collision shulkers.
fn sync_colliders(server: &Server) {
    let now = now_tick();
    let sync_tick = now % COLLIDER_SYNC_EVERY == 0;

    // Join tracking runs every tick for players that are not cleared yet, so that a player who joins first
    // and gets a body spawned near them later still counts as "joined a while ago", not "just now".
    // Cleared players only matter on sync ticks, so their (host-call heavy) checks are skipped in between.
    let all_players = server.get_all_players();
    let mut players = Vec::new();
    for pl in &all_players {
        let e = pl.as_entity();
        let pid = e.get_id() as u32;
        if !sync_tick && FIRST_SEEN.with(|f| f.borrow().get(&pid).is_some_and(|i| i.ready)) { continue; }
        let wname = pl.get_world().get_name();
        // A world change (respawn, dimension travel) makes the client reload everything, so it gets the
        // grace again. A client that is still loading cannot send movement, so the moment the player's
        // position or view changes from one tick to the next, it is certainly in the world and the
        // grace ends early.
        let pos = e.get_position();
        let look = (e.get_yaw() as f32, e.get_pitch() as f32);
        let ready = FIRST_SEEN.with(|f| {
            let mut f = f.borrow_mut();
            let ent = f.entry(pid).or_insert_with(|| JoinInfo { first: now, world: wname.clone(), pos, look, ready: false });
            if ent.world != wname {
                *ent = JoinInfo { first: now, world: wname.clone(), pos, look, ready: false };
            }
            if !ent.ready {
                // Compare with where the player was last tick, then remember the current state.
                let changed = dist_sq(pos, ent.pos) > 1e-4
                    || (look.0 - ent.look.0).abs() > 0.01
                    || (look.1 - ent.look.1).abs() > 0.01;
                ent.pos = pos;
                ent.look = look;
                if changed || now.saturating_sub(ent.first) >= JOIN_GRACE_TICKS { ent.ready = true; }
            }
            ent.ready
        });
        let Some(jp) = pl.as_java() else { continue };
        if !ready { continue; }
        players.push((pid, wname, e.get_eye_position(), jp));
    }
    if !sync_tick { return; }

    // Forget players that left, so a reconnect (possibly with a reused id) gets the grace again.
    let online: HashSet<u32> = all_players.iter().map(|pl| pl.as_entity().get_id() as u32).collect();
    FIRST_SEEN.with(|f| f.borrow_mut().retain(|k, _| online.contains(k)));

    let grave: Vec<Rc<Collider>> = GRAVE.with(|g| std::mem::take(&mut *g.borrow_mut()));
    let bodies: Vec<(String, Rc<BodyVis>)> = STATE.with(|s| s.borrow().bodies.iter().map(|((w, _), v)| (w.clone(), v.clone())).collect());
    if grave.is_empty() && bodies.is_empty() { return; }

    for c in &grave {
        for (pid, _) in c.viewers.borrow().iter() {
            if let Some((_, _, _, jp)) = players.iter().find(|p| p.0 == *pid) { jp.send_packet(&remove_packet(c)); }
        }
    }
    if players.is_empty() { return; }

    // Clients forget entities on respawn; re-announce now and then.
    let resend = now % (COLLIDER_SYNC_EVERY * 100) == 0;
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
                    // shulker there; announce it again a few times (windows, as we only run every 2nd tick).
                    (true, Some(i)) if matches!(now.saturating_sub(viewers[i].1), 3..=4 | 10..=11 | 40..=41) => {
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
            if !viewers.is_empty() {
                viewers.retain(|v| players.iter().any(|p| p.0 == v.0 && &p.1 == wname));
            }
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
    let did = display.get_id();
    OWN_IDS.with(|o| { o.borrow_mut().insert(did); });

    // Collision (full-cube blocks only; a shulker is a full cube, slabs/stairs would feel wrong): see `Collider`.
    let feet = (center.0, center.1 - 0.5, center.2);
    let col = solid.then(|| {
        let a = world.spawn_entity(EntityType::BlockDisplay, feet);
        if let Some(abd) = a.as_block_display() { abd.get_display().set_teleport_duration(2); }
        a.set_has_gravity(false);
        tag(&a);
        OWN_IDS.with(|o| { o.borrow_mut().insert(a.get_id()); });
        Rc::new(Collider { eid: a.get_id() as i32, anchor: a, target: StdCell::new(feet), viewers: RefCell::new(Vec::new()) })
    });
    Ok(PartVis { id, did, offset, state, display, bd, col, nbt, inv: RefCell::new(Vec::new()), store: RefCell::new(None), strikes: StdCell::new(0) })
}

thread_local! {
    /// Anchors of removed colliders, kept alive briefly: a click packet for the client-side shulker can
    /// still be in flight, and Pumpkin panics ("entity not found") if the entity is already gone.
    static PENDING_ANCHORS: RefCell<Vec<(Rc<Collider>, i64)>> = const { RefCell::new(Vec::new()) };
}
const ANCHOR_LINGER_MS: i64 = 1500;

fn flush_pending_anchors() {
    if PENDING_ANCHORS.with(|p| p.borrow().is_empty()) { return; }
    let now = now_millis();
    let due: Vec<Rc<Collider>> = PENDING_ANCHORS.with(|p| {
        let mut p = p.borrow_mut();
        let mut keep = Vec::new();
        let mut out = Vec::new();
        for (c, t) in p.drain(..) {
            if t <= now { out.push(c); } else { keep.push((c, t)); }
        }
        *p = keep;
        out
    });
    for c in due {
        c.anchor.remove();
        OWN_IDS.with(|o| { o.borrow_mut().remove(&(c.eid as u32)); });
    }
}

fn kill_part(p: PartVis) {
    p.display.remove();
    OWN_IDS.with(|o| { o.borrow_mut().remove(&p.did); });
    if let Some(c) = p.col {
        PENDING_ANCHORS.with(|a| a.borrow_mut().push((c.clone(), now_millis() + ANCHOR_LINGER_MS)));
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

/// Bodies further apart than this (blocks, on any axis) are not served by one shared entity query.
const MAX_QUERY_SPAN: f64 = 96.0;

struct PushBody {
    lo: [f64; 3],
    hi: [f64; 3],
    pos: [f64; 3],
    rot: [f32; 4],
    vis: Rc<BodyVis>,
    /// Block offsets, built the first time something has to be pushed out of this body.
    offsets: Option<Vec<[i32; 3]>>,
}

/// Finds the living entities (and items) near the bodies of a world. Mobs and items are pushed out of the
/// blocks here; everything living is reported to the engine so bodies can push / be stood on.
///
/// Every host call is a round trip, so this is built to make as few as possible: one entity query for a
/// whole group of bodies (not one per body), our own displays and anchors skipped by id right away, and
/// only the remaining entities asked for their type / position.
fn collect_pushers(name: &str) -> Vec<PusherInput> {
    let mut bodies: Vec<PushBody> = STATE.with(|s| {
        let st = s.borrow();
        let Some(e) = st.engines.get(name) else { return Vec::new() };
        e.ids().into_iter().filter_map(|id| {
            let (lo, hi) = e.body_bounds(id)?;
            let (pos, rot, _) = e.body_state(id)?;
            let vis = st.bodies.get(&(name.to_string(), id))?.clone();
            Some(PushBody { lo, hi, pos, rot, vis, offsets: None })
        }).collect()
    });
    if bodies.is_empty() { return Vec::new(); }

    // One group for everything unless the bodies are spread far apart; then each body queries alone.
    let mut span_lo = [f64::MAX; 3];
    let mut span_hi = [f64::MIN; 3];
    for b in &bodies {
        for i in 0..3 { span_lo[i] = span_lo[i].min(b.lo[i]); span_hi[i] = span_hi[i].max(b.hi[i]); }
    }
    let together = (0..3).all(|i| span_hi[i] - span_lo[i] <= MAX_QUERY_SPAN);
    let groups: Vec<Vec<usize>> = if together { vec![(0..bodies.len()).collect()] } else { (0..bodies.len()).map(|i| vec![i]).collect() };

    let mut seen: HashSet<u32> = HashSet::new();
    let mut out = Vec::new();
    for group in groups {
        // Box that covers the group (with the same margins the per-entity test below uses).
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for &i in &group {
            for k in 0..3 { lo[k] = lo[k].min(bodies[i].lo[k] - 2.5); hi[k] = hi[k].max(bodies[i].hi[k] + 2.5); }
        }
        // The query is centred on one of the group's displays.
        let nearby: Vec<_> = 'q: {
            for &i in &group {
                let Ok(parts) = bodies[i].vis.parts.try_borrow() else { continue };
                let Some(first) = parts.first() else { continue };
                let c = first.display.get_position();
                let half = |k: usize, c: f64| (c - lo[k]).abs().max((hi[k] - c).abs()) + 1.0;
                break 'q first.display.get_nearby_entities(half(0, c.0), half(1, c.1), half(2, c.2)).into_iter().collect();
            }
            Vec::new()
        };

        for e in nearby {
            let id = e.get_id();
            if OWN_IDS.with(|o| o.borrow().contains(&id)) || !seen.insert(id) { continue; }
            let living = e.is_living();
            let ty = e.get_type();
            if !living && !matches!(ty, EntityType::Item) { continue; }
            let p = e.get_position();
            let (w, h) = (e.get_width(), e.get_height());
            let mut near_any = false;
            for &i in &group {
                let b = &mut bodies[i];
                let near = p.0 > b.lo[0] - 2.0 && p.0 < b.hi[0] + 2.0 && p.1 > b.lo[1] - 2.5 && p.1 < b.hi[1] + 2.0 && p.2 > b.lo[2] - 2.0 && p.2 < b.hi[2] + 2.0;
                if !near { continue; }
                near_any = true;
                // Mobs and items can't walk through the blocks: push them out (players have the client-side shulkers).
                if !matches!(ty, EntityType::Player) {
                    if b.offsets.is_none() {
                        let Ok(parts) = b.vis.parts.try_borrow() else { continue };
                        b.offsets = Some(parts.iter().map(|p| p.offset).collect());
                    }
                    if let Some(offs) = &b.offsets {
                        push_out(&e, p, f64::from(w) * 0.5, f64::from(h), b.pos, b.rot, offs);
                    }
                }
            }
            if near_any && living {
                out.push(PusherInput { key: u64::from(id), feet: [p.0, p.1, p.2], half_width: w * 0.5, height: h });
            }
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
fn recover(server: &Server) -> bool {
    let session = session();
    // Set when the next scan should come soon: a display is on strike, or a saved body waits for parts.
    let mut again = false;

    // Entity ids of every helper entity we currently own.
    let mut live: HashSet<u32> = HashSet::new();
    PENDING_ANCHORS.with(|p| { for (c, _) in p.borrow().iter() { live.insert(c.anchor.get_id()); } });
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
            // Everything this plugin spawns is a block display. Checking the type first costs one host call
            // for the (usually vast majority of) other entities instead of two.
            if !matches!(e.get_type(), EntityType::BlockDisplay) { continue; }
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
                // The chunk lookup creates a host resource, so it only happens for displays that look missing.
                let missing = !loaded.contains(&p.did) && {
                    let c = part_center(pos, rot, p.offset);
                    world.get_chunk((c.0.floor() as i32) >> 4, (c.2.floor() as i32) >> 4).is_some()
                };
                if missing {
                    p.strikes.set(p.strikes.get().saturating_add(1));
                    // A copy of this body just reappeared, or the display has been missing for two scans.
                    if copy_seen || p.strikes.get() >= 2 { need = true; } else { again = true; }
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
    again || STATE.with(|s| !s.borrow().waiting.is_empty())
}

/// Whether an entity scan can be worth its cost right now (see `SCAN_MIN_GAP_TICKS`).
fn scan_due(now: u64) -> bool {
    STATE.with(|s| {
        let st = s.borrow();
        let since = now.saturating_sub(st.last_scan);
        st.last_scan == 0
            || (st.scan_again && since >= RECOVER_EVERY_TICKS)
            || (st.chunks_dirty && since >= SCAN_MIN_GAP_TICKS)
            || since >= SCAN_IDLE_TICKS
    })
}

/// Notes when a player moved to another chunk (or joined / left): that is when chunks, and the saved
/// entities in them, may have loaded. Two host calls per player, no entity enumeration.
fn watch_players(server: &Server) {
    let mut now: Vec<(u32, i32, i32)> = server.get_all_players().iter().map(|p| {
        let e = p.as_entity();
        let pos = e.get_position();
        (e.get_id(), (pos.0.floor() as i32) >> 4, (pos.2.floor() as i32) >> 4)
    }).collect();
    now.sort_unstable();
    STATE.with(|s| {
        let mut st = s.borrow_mut();
        if st.player_chunks != now { st.chunks_dirty = true; st.player_chunks = now; }
    });
}

fn housekeeping(server: &Server) {
    let now = now_tick();
    if now % 20 == 3 { watch_players(server); }
    if now % RECOVER_EVERY_TICKS == 1 && scan_due(now) {
        let again = recover(server);
        STATE.with(|s| { let mut st = s.borrow_mut(); st.last_scan = now; st.scan_again = again; st.chunks_dirty = false; });
    }
    if now % REFRESH_EVERY_TICKS == 0 { refresh_tags(false); sync_storage(); }
    if now % RECOVER_EVERY_TICKS == 2 {
        // Forget crouch state of players that left.
        let online: HashSet<u32> = server.get_all_players().iter().map(|p| p.as_entity().get_id()).collect();
        STATE.with(|s| s.borrow_mut().sneaking.retain(|id| online.contains(id)));
    }
}

// ---------------------------------------------------------------------------
// profiler (`/physics profile`)
// ---------------------------------------------------------------------------

const PROF_HOUSE: usize = 0;
const PROF_TNT: usize = 1;
const PROF_GRAB: usize = 2;
const PROF_PUSH: usize = 3;
const PROF_ENGINE: usize = 4;
const PROF_VISUAL: usize = 5;
const PROF_COLLIDER: usize = 6;
const PROF_TOTAL: usize = 7;
const PROF_N: usize = 8;
const PROF_NAMES: [&str; PROF_N] = ["housekeep", "blasts", "grabs", "pushers", "engine", "visuals", "colliders", "TOTAL"];

#[derive(Default)]
struct Prof { cur: [f64; PROF_N], total: [f64; PROF_N], max: [f64; PROF_N], ticks: u32 }
thread_local! { static PROF: RefCell<Prof> = RefCell::new(Prof::default()); }

/// Adds the time since `*t` to a section and restarts the stopwatch.
fn lap(slot: usize, t: &mut Instant) {
    let n = Instant::now();
    let ms = n.duration_since(*t).as_secs_f64() * 1000.0;
    PROF.with(|p| p.borrow_mut().cur[slot] += ms);
    *t = n;
}
fn prof_end_tick(start: Instant) {
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    PROF.with(|p| {
        let mut p = p.borrow_mut();
        p.cur[PROF_TOTAL] = ms;
        for i in 0..PROF_N {
            p.total[i] += p.cur[i];
            if p.cur[i] > p.max[i] { p.max[i] = p.cur[i]; }
            p.cur[i] = 0.0;
        }
        p.ticks += 1;
    });
}

fn tick(server: Server) {
    let start = Instant::now();
    let mut t = start;
    STATE.with(|s| s.borrow_mut().ticks += 1);
    housekeeping(&server);
    flush_pending_anchors();
    lap(PROF_HOUSE, &mut t);
    process_blasts(&server);
    lap(PROF_TNT, &mut t);
    apply_grabs(&server);
    lap(PROF_GRAB, &mut t);
    let names: Vec<String> = STATE.with(|s| s.borrow().engines.iter().filter(|(_, e)| !e.is_empty()).map(|(n, _)| n.clone()).collect());
    for name in names {
        let Some(world) = server.get_world_by_name(&name) else { continue };
        let pushers = collect_pushers(&name);
        lap(PROF_PUSH, &mut t);
        let updates = STATE.with(|s| {
            let mut st = s.borrow_mut();
            st.engines.get_mut(&name).map(|e| e.tick(&mut View::new(&world), &pushers))
        }).unwrap_or_default();
        lap(PROF_ENGINE, &mut t);

        // Resolve every update to its visuals under one borrow (and one key allocation per world).
        let resolved: Vec<_> = STATE.with(|s| {
            let st = s.borrow();
            let mut key: Option<(String, BodyId)> = None;
            updates.into_iter().map(|u| {
                let k = key.get_or_insert_with(|| (name.clone(), u.id));
                k.1 = u.id;
                let vis = st.bodies.get(&*k).cloned();
                (u, vis)
            }).collect()
        });

        for (u, vis) in resolved {
            let Some(vis) = vis else { continue };
            if u.removed {
                STATE.with(|s| { s.borrow_mut().bodies.remove(&(name.clone(), u.id)); });
                let parts: Vec<PartVis> = vis.parts.borrow_mut().drain(..).collect();
                for p in parts {
                    kill_part(p);
                }
                continue;
            }
            // Resting (or barely moving) bodies cost nothing: no teleports, no packets. The comparisons are
            // against what was last *applied*, so slow drift still gets through once it adds up.
            let old = vis.last_rot.get();
            let dot = old[0] * u.rot[0] + old[1] * u.rot[1] + old[2] * u.rot[2] + old[3] * u.rot[3];
            let rot_changed = 1.0 - dot.abs() > 1e-7;
            let lp = vis.last_pos.get();
            // Written so that a NaN (never placed yet) counts as moved.
            let still = (u.pos[0] - lp[0]).abs() <= POS_EPS && (u.pos[1] - lp[1]).abs() <= POS_EPS && (u.pos[2] - lp[2]).abs() <= POS_EPS;
            if still && !rot_changed { continue; }
            if rot_changed { vis.last_rot.set(u.rot); }
            vis.last_pos.set(u.pos);
            if let Ok(parts) = vis.parts.try_borrow() {
                for p in parts.iter() {
                    place_part_entities(p, u.pos, u.rot, rot_changed);
                }
            }
        }
        lap(PROF_VISUAL, &mut t);
    }
    sync_colliders(&server);
    lap(PROF_COLLIDER, &mut t);
    prof_end_tick(start);
}

// ---------------------------------------------------------------------------
// registering bodies
// ---------------------------------------------------------------------------

/// Spawns visuals + a body for a set of `(cell, state)` blocks. `anchor` is the cell of the offset-0 block.
fn spawn_body_rotated(world: &World, blocks: &[(Cell, u16)], anchor: Cell, spawn_center: Option<[f64; 3]>, rot: Option<[f32; 4]>, nbts: &[Option<Vec<u8>>]) -> Result<BodyId, String> {
    let wname = world.get_name();
    let mut view = View::new(world);
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

struct ProfileCmd;
impl CommandHandler for ProfileCmd {
    fn handle(&self, sender: CommandSender, server: Server, _a: ConsumedArgs) -> Result<i32, CommandError> {
        if !is_admin(&sender, &server) { return Err(CommandError::PermissionDenied); }
        let (ticks, total, max) = PROF.with(|p| {
            let mut p = p.borrow_mut();
            let r = (p.ticks, p.total, p.max);
            *p = Prof::default();
            r
        });
        if ticks == 0 {
            sender.send_message(TextComponent::text("No samples yet; run it again in a few seconds."));
            return Ok(0);
        }
        let (bodies, parts) = STATE.with(|s| {
            let st = s.borrow();
            (st.bodies.len(), st.bodies.values().map(|b| b.parts.try_borrow().map_or(0, |p| p.len())).sum::<usize>())
        });
        sender.send_message(TextComponent::text(&format!("physics: {ticks} ticks, {bodies} bodies, {parts} parts (ms per tick)")));
        for i in 0..PROF_N {
            sender.send_message(TextComponent::text(&format!("{:<10} avg {:>7.3}   max {:>7.3}", PROF_NAMES[i], total[i] / f64::from(ticks), max[i])));
        }
        Ok(ticks as i32)
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
        } else if RECENT_DEAD.with(|r| r.borrow().contains(&ev.entity_id)) {
            // Late click on a shulker whose block just went away: swallow it, don't act on what's behind.
            ev.cancelled = true;
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

    // Flint and steel on a TNT block lights it (before anything else, so it works while crouching too).
    if try_ignite(player, &world, &wname) { return true; }

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

// ---------------------------------------------------------------------------
// TNT: ignite with flint and steel, and explosions that hit physics bodies
// ---------------------------------------------------------------------------

/// Explosion power of a primed TNT entity (vanilla: 4).
const TNT_POWER: f64 = 4.0;
/// Peak speed (blocks/second) a one-block body gets from a point-blank blast.
const BLAST_SPEED: f32 = 20.0;
/// Extra bodies weigh more: speed is divided by `1 + BLAST_MASS_FACTOR * parts`.
const BLAST_MASS_FACTOR: f32 = 0.05;
/// Chance falloff multiplier for the hardness of a block (obsidian etc. never break).
const BLAST_HARDNESS_FACTOR: f32 = 0.15;

thread_local! {
    /// Primed TNT entities seen on the previous tick: entity id -> (world, last position).
    /// When one disappears, it has exploded.
        static RNG: StdCell<u64> = const { StdCell::new(0x2545_F491_4F6C_DD1D) };
}

fn rand01() -> f32 {
    RNG.with(|r| {
        let mut x = r.get() ^ (now_millis() as u64);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        r.set(x | 1);
        ((x >> 40) as f32) / 16_777_216.0
    })
}

/// Removes one part (visual + physics) and registers any bodies the removal split off.
/// Returns the split-off `(new body, part ids)` groups, or `None` when the part wasn't there.
/// Part ids stay the same when a part moves to a split-off body.
fn remove_and_split(wname: &str, body: BodyId, part: u32, vis: &Rc<BodyVis>) -> Option<Vec<(BodyId, Vec<u32>)>> {
    let result = STATE.with(|s| s.borrow_mut().engines.get_mut(wname).map(|e| e.remove_part(body, part)))?;
    if matches!(result, Removed::NotFound) { return None; }
    let removed = {
        let mut parts = vis.parts.borrow_mut();
        parts.iter().position(|p| p.id == part).map(|i| parts.remove(i))
    };
    if let Some(p) = removed { kill_part(p); }
    vis.invalidate();
    let mut out = Vec::new();
    match result {
        Removed::BodyGone => {
            STATE.with(|s| { s.borrow_mut().bodies.remove(&(wname.to_string(), body)); });
        }
        Removed::Ok(splits) => {
            for sp in splits {
                out.push((sp.body, sp.parts.iter().copied().collect::<Vec<u32>>()));
                let moved: Vec<PartVis> = {
                    let mut parts = vis.parts.borrow_mut();
                    let mut m = Vec::new();
                    let mut i = 0;
                    while i < parts.len() {
                        if sp.parts.contains(&parts[i].id) { m.push(parts.remove(i)); } else { i += 1; }
                    }
                    m
                };
                STATE.with(|s| {
                    let mut st = s.borrow_mut();
                    for p in &moved { write_static(p, sp.body); }
                    st.bodies.insert((wname.to_string(), sp.body), Rc::new(BodyVis::new(moved, vis.last_rot.get())));
                });
            }
        }
        Removed::NotFound => {}
    }
    Some(out)
}

/// Spawns a primed TNT entity where a physics TNT block was.
fn spawn_primed_tnt(world: &World, center: (f64, f64, f64)) {
    world.spawn_entity(EntityType::Tnt, (center.0, center.1 - 0.5, center.2));
}

/// Right-click with flint and steel on a TNT physics block: the block disappears and a TNT entity takes its place.
fn try_ignite(player: &pumpkin_plugin_api::Player, world: &World, wname: &str) -> bool {
    let Some(stack) = player.get_item_in_hand(Hand::Right) else { return false };
    if norm(&stack.get_registry_key()) != "flint_and_steel" { return false; }
    let e = player.as_entity();
    let (eye, dir) = (e.get_eye_position(), look_dir(&e));
    let Some(hit) = STATE.with(|s| s.borrow().engines.get(wname).and_then(|eng| eng.raycast([eye.0, eye.1, eye.2], dir, REACH))) else { return false };
    let Some(vis) = STATE.with(|s| s.borrow().bodies.get(&(wname.to_string(), hit.body)).cloned()) else { return false };
    let info = vis.parts.borrow().iter().find(|p| p.id == hit.part).map(|p| (p.offset, p.state));
    let Some((off, state)) = info else { return false };
    if !world::block_state_to_info(state).is_some_and(|i| norm(&i.name) == "tnt") { return false; }
    let Some((pos, rot, _)) = STATE.with(|s| s.borrow().engines.get(wname).and_then(|eng| eng.body_state(hit.body))) else { return true };
    let center = part_center(pos, rot, off);
    if remove_and_split(wname, hit.body, hit.part, &vis).is_none() { return true; }
    spawn_primed_tnt(world, center);
    world.play_custom_sound("minecraft:item.flintandsteel.use", SoundCategory::Blocks, center, 1.0, 1.0);
    true
}

thread_local! {
    /// Explosions the server announced (position, power) that still have to be applied to the bodies.
    static BLASTS: RefCell<Vec<((f64, f64, f64), f32)>> = const { RefCell::new(Vec::new()) };
}

/// The server announces every explosion (primed TNT, creepers, TNT minecarts, ...) just before it
/// happens. We only note it down; the bodies are pushed in the next `tick`, outside the explosion code.
/// No polling of the entity list is needed to find out about explosions.
struct ExplodeHandler;
impl EventHandler<EntityExplodeEvent> for ExplodeHandler {
    fn handle(&self, _server: Server, ev: EntityExplodeEventData) -> EntityExplodeEventData {
        // Touches only `BLASTS`: events can arrive while the plugin is in the middle of a host call, with
        // `STATE` already borrowed, so it must not be read here.
        if !ev.cancelled {
            BLASTS.with(|b| {
                let mut b = b.borrow_mut();
                if b.len() < 64 { b.push((ev.position, ev.yield_rate)); }
            });
        }
        ev
    }
}

/// Applies the explosions queued by [`ExplodeHandler`].
fn process_blasts(server: &Server) {
    let blasts = BLASTS.with(|b| std::mem::take(&mut *b.borrow_mut()));
    if blasts.is_empty() { return; }
    let active: Vec<String> = STATE.with(|s| s.borrow().engines.iter().filter(|(_, e)| !e.is_empty()).map(|(n, _)| n.clone()).collect());
    for (at, power) in blasts {
        // The event does not say which world it happened in; `explode_at` only touches bodies that are
        // actually within reach of the blast, so worlds without any are untouched.
        for wname in &active {
            if let Some(world) = server.get_world_by_name(wname) {
                explode_at(&world, wname, at, f64::from(power).clamp(1.0, 12.0));
            }
        }
    }
}

/// Applies an explosion to every physics body in range: shoves it away (off-centre hits also spin it),
/// then breaks parts with a chance that falls off with distance and rises with how soft the block is.
/// Real blocks between the blast and a part shield it. Other TNT blocks are set off.
fn explode_at(world: &World, wname: &str, at: (f64, f64, f64), power: f64) {
    let reach = power * 2.0;
    let bodies: Vec<(BodyId, [f64; 3], [f32; 4], Rc<BodyVis>)> = STATE.with(|s| {
        let st = s.borrow();
        let Some(e) = st.engines.get(wname) else { return Vec::new() };
        e.ids().into_iter().filter_map(|id| {
            let (lo, hi) = e.body_bounds(id)?;
            let (pos, rot, _) = e.body_state(id)?;
            let inside = at.0 > lo[0] - reach && at.0 < hi[0] + reach
                && at.1 > lo[1] - reach && at.1 < hi[1] + reach
                && at.2 > lo[2] - reach && at.2 < hi[2] + reach;
            if !inside { return None; }
            Some((id, pos, rot, st.bodies.get(&(wname.to_string(), id))?.clone()))
        }).collect()
    });

    // Where each part currently lives; changes when a removal splits the body.
    let mut loc: HashMap<(BodyId, u32), (BodyId, Rc<BodyVis>)> = HashMap::new();
    // (body at blast time, part id, centre, state, ignite?)
    let mut doomed: Vec<(BodyId, u32, (f64, f64, f64), u16, bool)> = Vec::new();

    for (bid, pos, rot, vis) in &bodies {
        let infos: Vec<(u32, [i32; 3], u16, bool)> = {
            let Ok(ps) = vis.parts.try_borrow() else { continue };
            ps.iter().map(|p| { snapshot(p); (p.id, p.offset, p.state, !p.inv.borrow().is_empty()) }).collect()
        };
        let n = infos.len() as f32;
        let mut nearest: Option<(f64, [i32; 3], (f64, f64, f64), bool)> = None;

        for (pid, off, state, has_items) in &infos {
            loc.insert((*bid, *pid), (*bid, vis.clone()));
            let c = part_center(*pos, *rot, *off);
            let d = dist_sq(c, at).sqrt();
            if d > reach { continue; }
            let shielded_by_world = world.ray_trace_blocks(at, c).is_some_and(|b| dist_sq(b, at).sqrt() + 0.05 < d);
            if nearest.as_ref().is_none_or(|b| d < b.0) { nearest = Some((d, *off, c, shielded_by_world)); }
            if shielded_by_world { continue; }

            let is_tnt = world::block_state_to_info(*state).is_some_and(|i| norm(&i.name) == "tnt");
            if is_tnt && d < reach * 0.75 {
                doomed.push((*bid, *pid, c, *state, true));
                continue;
            }
            if *has_items { continue; } // chests with contents survive (the items would be lost)
            let hardness = world::get_block_state_by_id(*state).map_or(1.0, |s| s.hardness);
            if hardness < 0.0 || hardness >= 15.0 { continue; }
            let falloff = (1.0 - d / (power * 1.15)).clamp(0.0, 1.0) as f32;
            let chance = falloff / (1.0 + hardness * BLAST_HARDNESS_FACTOR);
            if rand01() < chance { doomed.push((*bid, *pid, c, *state, false)); }
        }

        // Push the body away from the blast, from the part closest to it.
        if let Some((d, off, c, blocked)) = nearest {
            let mut impact = (1.0 - d / reach).clamp(0.0, 1.0) as f32;
            if blocked { impact *= 0.25; }
            let speed = BLAST_SPEED * (power / TNT_POWER) as f32 * impact / (1.0 + BLAST_MASS_FACTOR * n);
            if speed > 0.2 {
                let mut v = [(c.0 - at.0) as f32, (c.1 - at.1) as f32 + 0.4, (c.2 - at.2) as f32];
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-4);
                v = [v[0] / len, v[1] / len, v[2] / len];
                let target = [c.0 + f64::from(v[0] * speed), c.1 + f64::from(v[1] * speed), c.2 + f64::from(v[2] * speed)];
                let local = [off[0] as f32, off[1] as f32, off[2] as f32];
                // One full-strength blend sets the body's velocity at that point; it then flies on its own.
                STATE.with(|s| {
                    if let Some(eng) = s.borrow_mut().engines.get_mut(wname) {
                        eng.drag(*bid, local, target, GRAB_GAIN, speed, 1.0);
                    }
                });
            }
        }
    }

    // Break parts. The body may split along the way, so follow each part to wherever it ended up.
    for (bid, pid, c, state, ignite) in doomed {
        let Some((cur_body, cur_vis)) = loc.get(&(bid, pid)).cloned() else { continue };
        let Some(splits) = remove_and_split(wname, cur_body, pid, &cur_vis) else { continue };
        for (new_body, ids) in splits {
            let Some(nv) = STATE.with(|s| s.borrow().bodies.get(&(wname.to_string(), new_body)).cloned()) else { continue };
            for id in ids {
                // Parts that moved to the new body (keys are original body + part id).
                for ((ob, opid), v) in loc.iter_mut() {
                    if *ob == bid && *opid == id { *v = (new_body, nv.clone()); }
                }
            }
        }
        if ignite {
            spawn_primed_tnt(world, c);
        } else if let Some(info) = world::block_state_to_info(state) {
            world.play_custom_sound(block_sounds(&info.name).1, SoundCategory::Blocks, c, 1.0, 0.8);
        }
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

    let mut view = View::new(world);
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
            .then(CommandNode::literal("assemble").execute(AssembleCmd))
            .then(CommandNode::literal("profile").execute(ProfileCmd));
        ctx.register_command(cmd, PERM_USE);
        ctx.register_event_handler::<PlayerToggleSneakEvent, _>(SneakHandler, EventPriority::Normal, false)?;
        ctx.register_event_handler::<PlayerAnimationEvent, _>(AnimHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<PlayerInteractEntityEvent, _>(EntityHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<PlayerInteractUnknownEntityEvent, _>(UnknownEntityHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<PlayerInteractEvent, _>(UseHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<EntityExplodeEvent, _>(ExplodeHandler, EventPriority::Lowest, true)?;
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
