//! Pumpkin glue: /physics commands (assemble, delete), block-display visuals, collisions,
//! breaking / placing blocks on physics objects.
//!
//! Layout:
//!  - `plugin.rs` (this file): constants, shared state, world access, the per-tick loop, body
//!    registration, `/physics` commands and the plugin entry point.
//!  - `plugin/visuals.rs`: block-display entities, client-side collision shulkers.
//!  - `plugin/persist.rs`: saving bodies into the display entities' data, recovery after restarts /
//!    chunk unloads, chest contents (de)serialisation.
//!  - `plugin/interact.rs`: player interaction (punch, place, open, drag), TNT and explosions.
//!
//! The submodules do `use super::*;`, so everything imported or defined here (including private items)
//! is visible to them, and what they expose with `pub(super)` is visible to this file and to each other.
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
    events_wit::{BlockPlaceEventData, EntityExplodeEventData, EntityInteractionAction, InteractAction, PlayerAnimationEventData, PlayerInteractEntityEventData, PlayerInteractEventData, PlayerInteractUnknownEntityEventData, PlayerToggleSneakEventData, VehicleDestroyEventData},
    permission::{Permission, PermissionDefault, PermissionLevel},
    scheduler::SchedulerExt,
    text::TextComponent,
    world::{self, BlockFlags, World},
};
use pumpkin_plugin_api::events::entity::entity_explode::EntityExplodeEvent;
use pumpkin_plugin_api::events::block::block_place::BlockPlaceEvent;
use pumpkin_plugin_api::events::vehicle::vehicle_destroy::VehicleDestroyEvent;
use pumpkin_plugin_api::block_entity;
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

mod interact;
mod persist;
mod registry_ids;
mod visuals;

use interact::*;
use persist::*;
use visuals::*;

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

// ---------------------------------------------------------------------------
// shared state
// ---------------------------------------------------------------------------

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
    /// Minecarts that carry blocks, by minecart entity id.
    carts: HashMap<u32, Rc<Cart>>,
    /// Per-player punch tracking on cart blocks: (cart, cell, ticks).
    cart_punches: HashMap<u32, (u32, [i32; 3], Vec<u64>)>,
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
        punches: HashMap::new(), grabs: HashMap::new(), sneaking: HashSet::new(), carts: HashMap::new(), cart_punches: HashMap::new(),
    });
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

fn now_millis() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as i64)
}
fn dist_sq(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)
}
fn is_sneaking(player: &pumpkin_plugin_api::Player) -> bool {
    let id = player.as_entity().get_id();
    STATE.with(|s| s.borrow().sneaking.contains(&id))
}
fn now_tick() -> u64 { STATE.with(|s| s.borrow().ticks) }

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

/// Does this block state get a client-side collision shulker? True for anything that blocks movement
/// like a solid block: at least half a cube of collision volume (slabs, stairs, chests, ...) or a full
/// block height (fences, walls, panes, bars, closed doors / trapdoors / gates). Open doors, trapdoors
/// and gates, and low things like carpets, do not.
fn is_full_block(info: &StateInfo, state: u16) -> bool {
    const EPS: f32 = 1e-3;
    if info.boxes.is_empty() { return false; }
    if world::block_state_to_info(state).is_some_and(|i| i.properties.iter().any(|(k, v)| k == "open" && v == "true")) {
        return false;
    }
    let volume: f32 = info.boxes.iter()
        .map(|b| (b[3] - b[0]).max(0.0) * (b[4] - b[1]).max(0.0) * (b[5] - b[2]).max(0.0))
        .sum();
    let top = info.boxes.iter().map(|b| b[4]).fold(0.0f32, f32::max);
    volume >= 0.5 - EPS || top >= 1.0 - EPS
}

// ---------------------------------------------------------------------------
// profiler (`/physics profile`)
// ---------------------------------------------------------------------------

const PROF_HOUSE: usize = 0;
const PROF_TNT: usize = 1;
const PROF_GRAB: usize = 2;
const PROF_ENGINE: usize = 3;
const PROF_VISUAL: usize = 4;
const PROF_COLLIDER: usize = 5;
const PROF_TOTAL: usize = 6;
const PROF_N: usize = 7;
const PROF_NAMES: [&str; PROF_N] = ["housekeep", "blasts", "grabs", "engine", "visuals", "colliders", "TOTAL"];

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

// ---------------------------------------------------------------------------
// per-tick update
// ---------------------------------------------------------------------------

fn tick(server: Server) {
    let start = Instant::now();
    let mut t = start;
    STATE.with(|s| s.borrow_mut().ticks += 1);
    housekeeping(&server);
    flush_pending_anchors();
    lap(PROF_HOUSE, &mut t);
    process_cart_breaks(&server);
    process_fill(&server);
    process_blasts(&server);
    lap(PROF_TNT, &mut t);
    apply_grabs(&server);
    lap(PROF_GRAB, &mut t);
    let names: Vec<String> = STATE.with(|s| s.borrow().engines.iter().filter(|(_, e)| !e.is_empty()).map(|(n, _)| n.clone()).collect());
    for name in names {
        let Some(world) = server.get_world_by_name(&name) else { continue };
        let updates = STATE.with(|s| {
            let mut st = s.borrow_mut();
            st.engines.get_mut(&name).map(|e| e.tick(&mut View::new(&world)))
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
    cart_tick(&server);
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
            solid.push(is_full_block(&info, *state));
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

/// Saved contents of the chest-like block at `c`, read through the server's own container.
fn storage_contents(world: &World, c: Cell) -> Vec<u8> {
    use pumpkin_plugin_api::block_entity::BlockEntityType as Be;
    let inv = match world.get_block_entity(BlockPos { x: c[0], y: c[1], z: c[2] }) {
        Some(Be::ChestBlockEntity(b)) => b.get_container().get_inventory(),
        Some(Be::TrappedChestBlockEntity(b)) => b.get_container().get_inventory(),
        Some(Be::BarrelBlockEntity(b)) => b.get_container().get_inventory(),
        Some(Be::HopperBlockEntity(b)) => b.get_container().get_inventory(),
        Some(Be::DropperBlockEntity(b)) => b.get_container().get_inventory(),
        Some(Be::DispenserBlockEntity(b)) => b.get_container().get_inventory(),
        Some(Be::ShulkerBoxBlockEntity(b)) => b.get_container().get_inventory(),
        _ => return Vec::new(),
    };
    encode_inv(&inv)
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
        // Chest-like blocks: read their contents now (the block is removed below) and keep them as saved contents.
        let invs: Vec<Vec<u8>> = order.iter().map(|(c, state)| {
            let storage = world::block_state_to_info(*state).is_some_and(|b| matches!(interaction(&b.name), Some(Interaction::Storage(..))));
            if storage { storage_contents(&world, *c) } else { Vec::new() }
        }).collect();
        let bid = spawn_body_rotated(&world, &order, start, None, None, &nbts).map_err(|e| fail(&e))?;
        STATE.with(|s| {
            let st = s.borrow();
            if let Some(vis) = st.bodies.get(&(world.get_name(), bid)) {
                for p in vis.parts.borrow().iter() {
                    if let Some(inv) = invs.get(p.id as usize).filter(|v| !v.is_empty()) {
                        *p.inv.borrow_mut() = inv.clone();
                        write_static(p, bid);
                    }
                }
            }
        });
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
        // Only handlers that cancel or change their event are registered as blocking (the server waits for
        // them, usually while holding its own locks, and they call back into the server). Handlers that just
        // take note (swings, explosions, spawns) are not, so they can never hold the server up.
        ctx.register_event_handler::<PlayerToggleSneakEvent, _>(SneakHandler, EventPriority::Normal, false)?;
        ctx.register_event_handler::<PlayerAnimationEvent, _>(AnimHandler, EventPriority::Normal, false)?;
        ctx.register_event_handler::<PlayerInteractEntityEvent, _>(EntityHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<PlayerInteractUnknownEntityEvent, _>(UnknownEntityHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<PlayerInteractEvent, _>(UseHandler, EventPriority::Normal, true)?;
        ctx.register_event_handler::<BlockPlaceEvent, _>(PlaceFillHandler, EventPriority::Normal, false)?;
        ctx.register_event_handler::<EntityExplodeEvent, _>(ExplodeHandler, EventPriority::Lowest, false)?;
        ctx.register_event_handler::<VehicleDestroyEvent, _>(CartDestroyHandler, EventPriority::Normal, true)?;
        // No EntitySpawnEvent handler on purpose: every spawn_entity() from inside the plugin's own handlers would
        // make the server fire it back into the busy plugin and wait for it -> deadlock of the whole server.
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
            st.carts.clear();
        });
        Ok(())
    }
}
register_plugin!(Physics);
