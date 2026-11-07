//! Visuals: block-display entities per part, and client-side collision shulkers.
use super::*;

// ---------------------------------------------------------------------------
// visual data model
// ---------------------------------------------------------------------------

pub(super) struct PartVis {
    pub(super) id: u32,
    /// Entity id of `display` (cached: asking the host costs a round trip).
    pub(super) did: u32,
    pub(super) offset: [i32; 3],
    pub(super) state: u16,
    pub(super) display: Entity,
    pub(super) bd: BlockDisplayEntity,
    /// Collision for full-cube blocks (see [`Collider`]).
    pub(super) col: Option<Rc<Collider>>,
    /// Raw NBT of the block entity this block had (chest contents etc.). Inert while it is a physics block.
    pub(super) nbt: Option<Vec<u8>>,
    /// Contents of a chest-like block (see `snapshot`): `[slot, count, key length, key...]` per stack.
    pub(super) inv: RefCell<Vec<u8>>,
    /// Inventory of the GUI that was opened last; read back to keep `inv` current.
    pub(super) store: RefCell<Option<Inventory>>,
    /// Consecutive recovery scans in which this part's display was missing although its chunk is loaded.
    pub(super) strikes: StdCell<u8>,
}
pub(super) struct BodyVis {
    pub(super) parts: RefCell<Vec<PartVis>>,
    pub(super) last_rot: StdCell<[f32; 4]>,
    /// Part count last written into the entities' saved data (0 = must be rewritten).
    pub(super) n_saved: StdCell<usize>,
    /// Pose last written into the entities' saved data.
    pub(super) pose_saved: StdCell<Option<([f64; 3], [f32; 4])>>,
    /// Position the display entities were last teleported for (NaN = never, so the first update applies).
    pub(super) last_pos: StdCell<[f64; 3]>,
}
impl BodyVis {
    pub(super) fn new(parts: Vec<PartVis>, last_rot: [f32; 4]) -> Self {
        Self { parts: RefCell::new(parts), last_rot: StdCell::new(last_rot), n_saved: StdCell::new(0), pose_saved: StdCell::new(None), last_pos: StdCell::new([f64::NAN; 3]) }
    }
    /// Makes the next save rewrite the part count and pose.
    pub(super) fn invalidate(&self) {
        self.n_saved.set(0);
        self.pose_saved.set(None);
    }
}

// ---------------------------------------------------------------------------
// collision shulkers
// ---------------------------------------------------------------------------

/// Collision for one full-cube part: a shulker that exists only on clients (never in the world, so it
/// can't wander off, take damage or show up in the world data). It is announced to clients under the
/// entity id of the part's *anchor*, an invisible, empty block display at the block's bottom that does
/// exist on the server. That has three effects: the server's own position updates for the anchor move
/// the shulker on the client (so no per-tick packets of ours), Pumpkin finds a real entity when a
/// client punches the shulker (no "invalid entity" kick), and nothing extra has to be spawned.
pub(super) struct Collider {
    pub(super) anchor: Entity,
    /// The anchor's entity id, which the client-side shulker borrows.
    pub(super) eid: i32,
    /// Where the shulker's feet should be at spawn time.
    pub(super) target: StdCell<(f64, f64, f64)>,
    /// Players that currently have it spawned: (player entity id, tick sent).
    pub(super) viewers: RefCell<Vec<(u32, u64)>>,
    /// Tick the collider was created. The shulker is only announced a few ticks later (see `COLLIDER_SETTLE_TICKS`).
    pub(super) born: u64,
    /// `None`: a shulker (solid collision). `Some((width, height))`: a client-side Interaction entity of that
    /// size, used for blocks without a shulker so they can still be punched / used.
    pub(super) shape: Option<(f32, f32)>,
}
thread_local! {
    /// Removed colliders whose shulker still has to be despawned for their viewers.
    static GRAVE: RefCell<Vec<Rc<Collider>>> = const { RefCell::new(Vec::new()) };
    /// Ids of recently removed colliders: a client may still click the shulker for a moment.
    pub(super) static RECENT_DEAD: RefCell<VecDeque<i32>> = const { RefCell::new(VecDeque::new()) };
    /// Tick at which each player was first seen by `sync_colliders` (player entity id -> tick).
    static FIRST_SEEN: RefCell<HashMap<u32, JoinInfo>> = RefCell::new(HashMap::new());
}
/// Ticks a player must have been online before they get raw collider packets. A player that shows
/// up in the player list can still be finishing the join sequence (login / configuration / world
/// load); a spawn packet arriving then is a protocol violation and the client disconnects.
const JOIN_GRACE_TICKS: u64 = 200;
/// Earliest a joining / world-changing player can be treated as loaded, even if they seem to move.
const JOIN_MIN_TICKS: u64 = 60;
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
/// Protocol id of the interaction entity type.
const INTERACTION_TYPE_ID: i32 = 70;
fn spawn_packets(c: &Collider, at: (f64, f64, f64)) -> [ClientboundPacket; 2] {
    let n = c.eid as u64;
    // Entity flags (index 0, type byte): 0x20 = invisible. Interaction: width (index 8) and height (index 9) floats.
    let (r_type, metadata) = match c.shape {
        None => (SHULKER_TYPE_ID, vec![0, 0, 0x20, 0xFF]),
        Some((w, h)) => {
            let mut m = vec![8u8, 3];
            m.extend_from_slice(&w.to_be_bytes());
            m.extend_from_slice(&[9, 3]);
            m.extend_from_slice(&h.to_be_bytes());
            m.push(0xFF);
            (INTERACTION_TYPE_ID, m)
        }
    };
    [
        ClientboundPacket::CSpawnEntity(CSpawnEntity {
            entity_id: c.eid,
            entity_uuid: Uuid { high: 0xF4CE_5EED_0000_0000 | n, low: n },
            r_type,
            position: at,
            velocity: "[0.0,0.0,0.0]".to_string(),
            pitch: 0, yaw: 0, head_yaw: 0, data: 0,
        }),
        ClientboundPacket::CSetEntityMetadata(CSetEntityMetadata { entity_id: c.eid, metadata }),
    ]
}
fn remove_packet(c: &Collider) -> ClientboundPacket {
    ClientboundPacket::CRemoveEntities(CRemoveEntities { entity_ids: vec![c.eid] })
}

/// The collider proximity pass runs every this many ticks (spawn/despawn of client-side shulkers does
/// not need tick precision; the shulkers themselves are moved by the server's anchor teleports).
const COLLIDER_SYNC_EVERY: u64 = 2;
/// A brand-new anchor must be fully announced to the client (spawn + its display metadata) before our
/// shulker packets take over its entity id. If the server's own packets for the anchor arrive *after* the
/// shulker spawn, the client applies display metadata to a shulker and drops with a network protocol error
/// (timing-dependent, so it only happens sometimes). Waiting a few ticks makes the order deterministic.
const COLLIDER_SETTLE_TICKS: u64 = 6;

/// Keeps every nearby Java player's client in sync with the collision shulkers.
pub(super) fn sync_colliders(server: &Server) {
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
                let waited = now.saturating_sub(ent.first);
                // Never before JOIN_MIN_TICKS: the server moving the player (spawn teleport) looks like movement but
                // says nothing about the client having finished loading. A shulker spawn that reaches a client
                // without a level makes it crash with a network protocol error (NPE in Entity.<init>).
                if (changed && waited >= JOIN_MIN_TICKS) || waited >= JOIN_GRACE_TICKS { ent.ready = true; }
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
    let bodies: Vec<(String, Rc<BodyVis>)> = STATE.with(|s| {
        let st = s.borrow();
        st.bodies.iter().map(|((w, _), v)| (w.clone(), v.clone()))
            .chain(st.carts.values().map(|c| (c.world.clone(), c.vis.clone()))).collect()
    });
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
                    // Too fresh: the server hasn't finished announcing the anchor itself yet.
                    (true, None) if now.saturating_sub(c.born) < COLLIDER_SETTLE_TICKS => {}
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

// ---------------------------------------------------------------------------
// transforms and placing display entities
// ---------------------------------------------------------------------------

pub(super) fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
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

pub(super) fn part_center(pos: [f64; 3], rot: [f32; 4], offset: [i32; 3]) -> (f64, f64, f64) {
    let r = rotate(rot, [offset[0] as f32, offset[1] as f32, offset[2] as f32]);
    (pos[0] + f64::from(r[0]), pos[1] + f64::from(r[1]), pos[2] + f64::from(r[2]))
}

pub(super) fn place_part_entities(p: &PartVis, pos: [f64; 3], rot: [f32; 4], with_rotation: bool) {
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

pub(super) fn spawn_part(world: &World, state: u16, offset: [i32; 3], id: u32, center: (f64, f64, f64), solid: bool, nbt: Option<Vec<u8>>) -> Result<PartVis, String> {
    // Chest-like blocks keep their contents only in `inv` (saved / given as the `container` component). Block
    // entity NBT with its own `Items` would go stale as soon as the contents change and duplicate items.
    let nbt = if world::block_state_to_info(state).is_some_and(|i| matches!(interaction(&i.name), Some(Interaction::Storage(..)))) { None } else { nbt };
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
    // Blocks without a shulker get an Interaction entity sized to their outline instead.
    let shape = if solid { None } else {
        world::get_block_state_by_id(state).and_then(|s| {
            let b = &s.outline_shapes;
            if b.is_empty() { return None; }
            let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
            for x in b {
                for (i, (a, c)) in [(x.min.0, x.max.0), (x.min.1, x.max.1), (x.min.2, x.max.2)].into_iter().enumerate() {
                    lo[i] = lo[i].min(a); hi[i] = hi[i].max(c);
                }
            }
            Some((((hi[0] - lo[0]).max(hi[2] - lo[2])).max(0.1) as f32, ((hi[1] - lo[1]).max(0.1)) as f32))
        })
    };
    let col = (solid || shape.is_some()).then(|| {
        let a = world.spawn_entity(EntityType::BlockDisplay, feet);
        if let Some(abd) = a.as_block_display() { abd.get_display().set_teleport_duration(2); }
        a.set_has_gravity(false);
        tag(&a);
        OWN_IDS.with(|o| { o.borrow_mut().insert(a.get_id()); });
        Rc::new(Collider { eid: a.get_id() as i32, anchor: a, target: StdCell::new(feet), viewers: RefCell::new(Vec::new()), born: STATE.with(|s| s.try_borrow().map_or(0, |st| st.ticks)), shape })
    });
    Ok(PartVis { id, did, offset, state, display, bd, col, nbt, inv: RefCell::new(Vec::new()), store: RefCell::new(None), strikes: StdCell::new(0) })
}

thread_local! {
    /// Anchors of removed colliders, kept alive briefly: a click packet for the client-side shulker can
    /// still be in flight, and Pumpkin panics ("entity not found") if the entity is already gone.
    pub(super) static PENDING_ANCHORS: RefCell<Vec<(Rc<Collider>, i64)>> = const { RefCell::new(Vec::new()) };
}
const ANCHOR_LINGER_MS: i64 = 1500;

pub(super) fn flush_pending_anchors() {
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

pub(super) fn kill_part(p: PartVis) {
    p.display.remove();
    OWN_IDS.with(|o| { o.borrow_mut().remove(&p.did); });
    if let Some(c) = p.col {
        PENDING_ANCHORS.with(|a| a.borrow_mut().push((c.clone(), now_millis() + ANCHOR_LINGER_MS)));
        RECENT_DEAD.with(|r| { let mut r = r.borrow_mut(); r.push_back(c.eid); if r.len() > 256 { r.pop_front(); } });
        GRAVE.with(|g| g.borrow_mut().push(c));
    }
}
