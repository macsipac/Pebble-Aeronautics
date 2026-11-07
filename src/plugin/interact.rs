//! Player interaction with physics blocks: punching out, placing, opening chests, dragging, and
//! TNT / explosions.
use super::*;

// ---------------------------------------------------------------------------
// inventory helpers
// ---------------------------------------------------------------------------

/// Key of a chest's saved contents inside the custom-data of the item it was broken into.
const INV_ITEM_KEY: &[u8] = b"pm_inv";

/// Unnamed NBT compound `{<key>: [B; ...]}` (the custom-data component's wire format).
fn custom_data_with(key: &[u8], bytes: &[u8]) -> Vec<u8> {
    let mut v = vec![10u8, 7];
    v.extend_from_slice(&(key.len() as u16).to_be_bytes());
    v.extend_from_slice(key);
    v.extend_from_slice(&(bytes.len() as i32).to_be_bytes());
    v.extend_from_slice(bytes);
    v.push(0);
    v
}
/// Finds a byte array in serialized custom-data (searched for, so it doesn't matter whether the root
/// compound is written with or without a name).
fn custom_data_get(key: &[u8], v: &[u8]) -> Option<Vec<u8>> {
    let mut needle = vec![7u8];
    needle.extend_from_slice(&(key.len() as u16).to_be_bytes());
    needle.extend_from_slice(key);
    let at = v.windows(needle.len()).position(|w| w == needle.as_slice())? + needle.len();
    let len = i32::from_be_bytes(v.get(at..at + 4)?.try_into().ok()?);
    if len <= 0 { return None; }
    v.get(at + 4..at + 4 + len as usize).map(<[u8]>::to_vec)
}

/// The saved chest contents an item carries (see `part_item`), or empty.
fn held_inv(stack: &ItemStack) -> Vec<u8> {
    stack.get_components().into_iter().find(|c| is_comp(c.component, "custom-data")).and_then(|c| custom_data_get(INV_ITEM_KEY, &c.value)).unwrap_or_default()
}

/// One item for a broken part; it carries the block entity data (and, for a chest-like block, its saved
/// contents in custom-data, with a glint) when there is any.
fn part_item(name: &str, nbt: Option<&[u8]>, inv: &[u8]) -> ItemStack {
    let stack = ItemStack::of(name, 1);
    let (real, meta) = split_meta(inv);
    // The item's own components (custom name, ...) that the block was placed with.
    if let Some(m) = meta {
        for st in parse_inv(&join_meta(Vec::new(), Some(m))) {
            for (c, v) in &st.components { stack.set_component(*c, v); }
        }
    }
    if !real.is_empty() {
        if let Some(c) = comp_by_name("custom-data") { stack.set_component(c, &custom_data_with(INV_ITEM_KEY, &real)); }
        if let (Some(c), Some(v)) = (comp_by_name("container"), container_value(&real)) { stack.set_component(c, &v); }
        if let Some(c) = comp_by_name("enchantment-glint-override") { stack.set_component(c, &vec![1u8]); }
    }
    if let (Some(nbt), Some(comp)) = (nbt, comp_by_name(BLOCK_ENTITY_DATA)) {
        // Component wire format: a VarInt type id (0) followed by the unnamed NBT compound.
        let mut value = vec![0u8];
        value.extend_from_slice(nbt);
        stack.set_component(comp, &value);
    }
    stack
}


fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 { out.push(b); break; }
        out.push(b | 0x80);
    }
}

/// Wire value of the vanilla `container` component for saved chest contents: one entry per slot up to the last
/// used one, each `false` or `true` + item (registry id, count, component patch), exactly what
/// `container=[{slot:..,item:{id:..,count:..,components:{..}}}]` becomes. Slots whose item has no known
/// registry id are left out.
fn container_value(inv: &[u8]) -> Option<Vec<u8>> {
    let stacks: Vec<StoredStack> = parse_inv(inv).into_iter().filter(|s| s.slot != u32::from(META_SLOT) && registry_ids::item_id(&s.key).is_some()).collect();
    let len = stacks.iter().map(|s| s.slot + 1).max()?;
    let mut out = Vec::new();
    put_varint(&mut out, len);
    for slot in 0..len {
        let Some(st) = stacks.iter().find(|s| s.slot == slot) else { out.push(0); continue };
        out.push(1);
        put_varint(&mut out, registry_ids::item_id(&st.key)?);
        put_varint(&mut out, st.count);
        let comps: Vec<(u32, &Vec<u8>)> = st.components.iter()
            .filter_map(|(c, v)| Some((*c as u32, v)))
            .collect();
        put_varint(&mut out, comps.len() as u32);
        put_varint(&mut out, 0);
        for (id, v) in comps { put_varint(&mut out, id); out.extend_from_slice(v); }
    }
    Some(out)
}

/// For chest-like blocks: adds the placed item's own components (custom name, ...) to the saved contents.
fn with_item_meta(contents: Vec<u8>, stack: &ItemStack, state: u16) -> Vec<u8> {
    if !world::block_state_to_info(state).is_some_and(|i| matches!(interaction(&i.name), Some(Interaction::Storage(..)))) { return contents; }
    let comps: Vec<(DataComponent, Vec<u8>)> = stack.get_components().into_iter()
        .filter(|c| !["container", "block-entity-data", "custom-data", "enchantment-glint-override"].iter().any(|n| is_comp(c.component, n)))
        .map(|c| (c.component, c.value)).collect();
    if comps.is_empty() { return contents; }
    join_meta(contents, Some(encode_meta(&stack.get_registry_key(), &comps)))
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
pub(super) fn norm(key: &str) -> &str { key.strip_prefix("minecraft:").unwrap_or(key) }

/// Puts one item into the main inventory: onto a matching stack if there is room, else into a free slot.
/// Items with block-entity data or chest contents never stack.
fn give(player: &pumpkin_plugin_api::Player, name: &str, nbt: Option<&[u8]>, inv_bytes: &[u8]) {
    let inv = player.get_inventory().as_inventory();
    let size = inv.get_size().min(36);
    if nbt.is_none() && inv_bytes.is_empty() {
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
        inv.set_item(slot, Some(part_item(name, nbt, inv_bytes)));
    }
}

// ---------------------------------------------------------------------------
// normal placement of a chest item that carries contents
// ---------------------------------------------------------------------------
//
// Pumpkin applies an item's `container` component right after setting the block, but a chest only creates its
// block entity afterwards (in its own `placed` hook), so the component is dropped. The contents are therefore
// also written into the new block entity from the item's saved contents, as soon as it exists.

thread_local! {
    static FILL: RefCell<Vec<(String, [i32; 3], Vec<u8>, u8)>> = const { RefCell::new(Vec::new()) };
}

pub(super) struct PlaceFillHandler;
impl EventHandler<BlockPlaceEvent> for PlaceFillHandler {
    fn handle(&self, _server: Server, ev: BlockPlaceEventData) -> BlockPlaceEventData {
        if ev.cancelled || !ev.can_build { return ev; }
        if !matches!(interaction(&ev.block_placed), Some(Interaction::Storage(..))) { return ev; }
        for hand in [Hand::Right, Hand::Left] {
            let Some(stack) = ev.player.get_item_in_hand(hand) else { continue };
            if norm(&stack.get_registry_key()) != norm(&ev.block_placed) { continue; }
            let (real, _) = split_meta(&held_inv(&stack));
            if real.is_empty() { continue; }
            let wname = ev.player.get_world().get_name();
            FILL.with(|f| f.borrow_mut().push((wname, [ev.block_pos.x, ev.block_pos.y, ev.block_pos.z], real, 0)));
            break;
        }
        ev
    }
}

fn fill_container(world: &World, c: [i32; 3], inv: &[u8]) -> bool {
    use pumpkin_plugin_api::block_entity::BlockEntityType as Be;
    let container = match world.get_block_entity(BlockPos { x: c[0], y: c[1], z: c[2] }) {
        Some(Be::ChestBlockEntity(b)) => b.get_container(),
        Some(Be::TrappedChestBlockEntity(b)) => b.get_container(),
        Some(Be::BarrelBlockEntity(b)) => b.get_container(),
        Some(Be::HopperBlockEntity(b)) => b.get_container(),
        Some(Be::DropperBlockEntity(b)) => b.get_container(),
        Some(Be::DispenserBlockEntity(b)) => b.get_container(),
        Some(Be::ShulkerBoxBlockEntity(b)) => b.get_container(),
        _ => return false,
    };
    let size = container.get_size();
    for st in parse_inv(inv) {
        if st.slot < size && st.slot != u32::from(META_SLOT) { container.set_stack(st.slot, Some(stored_item(&st))); }
    }
    true
}

/// Called every tick: fills freshly placed chests once their block entity exists (retries for a few ticks).
pub(super) fn process_fill(server: &Server) {
    let pending = FILL.with(|f| std::mem::take(&mut *f.borrow_mut()));
    if pending.is_empty() { return; }
    let mut keep = Vec::new();
    for (w, pos, inv, tries) in pending {
        let Some(world) = server.get_world_by_name(&w) else { continue };
        if fill_container(&world, pos, &inv) { continue; }
        if tries < 10 { keep.push((w, pos, inv, tries + 1)); }
    }
    FILL.with(|f| f.borrow_mut().extend(keep));
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
/// Every part has a client-only collider (a shulker, or an Interaction entity for blocks that aren't full
/// cubes). Clicks on it arrive as [`EntityHandler`] events and name the exact block; the ray-cast in
/// [`UseHandler`] is the fallback.

/// True when `id` is the anchor entity whose id one of our client-only shulkers borrows.
fn is_collider(id: i32) -> bool {
    STATE.with(|s| s.borrow().bodies.values().any(|v| {
        v.parts.try_borrow().is_ok_and(|ps| ps.iter().any(|p| p.col.as_ref().is_some_and(|c| c.eid == id)))
    }))
}

/// Our client-only shulkers use the entity id of a real anchor, so Pumpkin finds the target of a punch
/// and doesn't kick for "attacking an invalid entity". Punches themselves are counted from the arm
/// swing; a right-click on the shulker is treated as a click on the block.
pub(super) struct EntityHandler;
impl EventHandler<PlayerInteractEntityEvent> for EntityHandler {
    fn handle(&self, _server: Server, mut ev: PlayerInteractEntityEventData) -> PlayerInteractEntityEventData {
        if is_collider(ev.entity_id) {
            ev.cancelled = true;
            if !matches!(ev.action, EntityInteractionAction::Attack) {
                // Same as for carts: the click names the exact shulker (= block), so open it directly.
                let opened = !is_sneaking(&ev.player) && try_open_by_collider(&ev.player, ev.entity_id, now_tick());
                if !opened { use_block(&ev.player); }
            }
        } else if is_cart_collider(ev.entity_id) {
            ev.cancelled = true;
            if !matches!(ev.action, EntityInteractionAction::Attack) {
                // The click names the exact shulker (= block) that was hit: open that block directly instead of
                // re-casting a ray on the server, which can miss at block edges and then silently does nothing.
                let now = now_tick();
                let opened = !is_sneaking(&ev.player) && cart_part_by_collider(ev.entity_id)
                    .is_some_and(|(cart, cell)| cart_try_open(&ev.player, &CartHit { cart, cell, normal: [0, 0, 0], is_cart: false }, now));
                if !opened { cart_use(&ev.player, None); }
            }
        } else if !matches!(ev.action, EntityInteractionAction::Attack) && cart_entity_click(&ev.player, ev.entity_id) {
            // A block in hand on a minecart: place it instead of sitting down.
            ev.cancelled = true;
        } else if RECENT_DEAD.with(|r| r.borrow().contains(&ev.entity_id)) {
            // Late click on a shulker whose block just went away: swallow it, don't act on what's behind.
            ev.cancelled = true;
        }
        ev
    }
}

/// Safety net: a click on a shulker whose anchor was removed a moment ago would get the player kicked
/// for an unknown entity; swallow it instead.
pub(super) struct UnknownEntityHandler;
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
pub(super) struct UseHandler;
impl EventHandler<PlayerInteractEvent> for UseHandler {
    fn handle(&self, _server: Server, mut ev: PlayerInteractEventData) -> PlayerInteractEventData {
        refresh_grab(&ev.player);
        let right = matches!(ev.action, InteractAction::RightClickAir | InteractAction::RightClickBlock);
        // A cart item (minecart / cart egg) used on a block: put the cart there ourselves, right now, instead of letting
        // Pumpkin spawn the placeholder zombie of the egg. (Spawning here is safe: there is no EntitySpawnEvent handler.)
        if matches!(ev.action, InteractAction::RightClickBlock) {
            if let Some(pos) = ev.clicked_pos {
                if try_place_cart(&ev.player, pos, ev.block.ends_with("rail")) { ev.cancelled = true; return ev; }
            }
        }
        // A block that rides on a cart: place onto it (right-click) or swallow the click (left-click punches).
        if cart_hit(&ev.player, REACH).is_some_and(|h| !h.is_cart) {
            if right { cart_use(&ev.player, None); }
            ev.cancelled = true;
            return ev;
        }
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
pub(super) enum Interaction { Storage(Screen, &'static str), EnderChest }
pub(super) fn interaction(name: &str) -> Option<Interaction> {
    let n = norm(name);
    match n {
        "chest" | "trapped_chest" => Some(Interaction::Storage(Screen::Generic9x3, "Chest")),
        "barrel" => Some(Interaction::Storage(Screen::Generic9x3, "Barrel")),
        "hopper" => Some(Interaction::Storage(Screen::Hopper, "Hopper")),
        "dropper" => Some(Interaction::Storage(Screen::Generic3x3, "Dropper")),
        "dispenser" => Some(Interaction::Storage(Screen::Generic3x3, "Dispenser")),
        "ender_chest" => Some(Interaction::EnderChest),
        _ if n.ends_with("shulker_box") => Some(Interaction::Storage(Screen::ShulkerBox, "Shulker Box")),
        _ => None,
    }
}

/// Opens the storage / ender chest GUI of one part. Shared by physics bodies and carts. True (the click is
/// ours) even when the debounce swallowed it.
fn open_part(player: &pumpkin_plugin_api::Player, p: &PartVis, kind: Interaction, now: u64) -> bool {
    let pid = player.as_entity().get_id();
    if STATE.with(|s| s.borrow().last_action.get(&pid).is_some_and(|t| now.saturating_sub(*t) < 8)) { return true; }
    STATE.with(|s| { s.borrow_mut().last_action.insert(pid, now); });
    let sound = match &kind {
        Interaction::EnderChest => Some("minecraft:block.ender_chest.open"),
        Interaction::Storage(_, "Chest") => Some("minecraft:block.chest.open"),
        Interaction::Storage(_, "Barrel") => Some("minecraft:block.barrel.open"),
        Interaction::Storage(_, "Shulker Box") => Some("minecraft:block.shulker_box.open"),
        Interaction::Storage(..) => None,
    };
    if let Some(snd) = sound {
        player.get_world().play_custom_sound(snd, SoundCategory::Blocks, p.display.get_position(), 0.5, 0.9 + 0.2 * rand01());
    }
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

/// The body whose client-side collider uses entity id `id`, and that part's id.
fn body_part_by_collider(id: i32) -> Option<(Rc<BodyVis>, u32)> {
    STATE.with(|s| {
        let st = s.try_borrow().ok()?;
        st.bodies.values().find_map(|v| {
            let ps = v.parts.try_borrow().ok()?;
            ps.iter().find(|p| p.col.as_ref().is_some_and(|c| c.eid == id)).map(|p| (v.clone(), p.id))
        })
    })
}

/// A click that names the exact collider (= block) that was hit: opens that block directly instead of
/// re-casting a ray on the server, which can miss at block edges and then silently does nothing.
fn try_open_by_collider(player: &pumpkin_plugin_api::Player, collider_id: i32, now: u64) -> bool {
    let Some((vis, part)) = body_part_by_collider(collider_id) else { return false };
    let parts = vis.parts.borrow();
    let Some(p) = parts.iter().find(|p| p.id == part) else { return false };
    let Some(kind) = world::block_state_to_info(p.state).and_then(|i| interaction(&i.name)) else { return false };
    open_part(player, p, kind, now)
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
    open_part(player, p, kind, now)
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

// ---------------------------------------------------------------------------
// dragging
// ---------------------------------------------------------------------------

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
pub(super) fn apply_grabs(server: &Server) {
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
pub(super) struct SneakHandler;
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

// ---------------------------------------------------------------------------
// punching / breaking
// ---------------------------------------------------------------------------

/// Every left-click swings the arm, whether it hits a block, an entity or nothing. We ray-cast
/// against the physics bodies ourselves; hitting the same block `PUNCHES_TO_BREAK` times within one
/// second breaks it.
pub(super) struct AnimHandler;
impl EventHandler<PlayerAnimationEvent> for AnimHandler {
    fn handle(&self, _server: Server, ev: PlayerAnimationEventData) -> PlayerAnimationEventData {
        if ev.animation_type == "ArmSwingMain" {
            punch(&ev.player);
        }
        ev
    }
}

fn punch(player: &pumpkin_plugin_api::Player) {
    if cart_punch(player) { return; }
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
    } else {
        // Feedback for every punch that doesn't break it yet: the block's hit sound, higher each time.
        let state = STATE.with(|s| s.borrow().bodies.get(&(wname.clone(), hit.body)).and_then(|v| {
            v.parts.try_borrow().ok().and_then(|ps| ps.iter().find(|p| p.id == hit.part).map(|p| p.state))
        }));
        if let Some(info) = state.and_then(world::block_state_to_info) {
            let at = (eye.0 + f64::from(dir[0]) * f64::from(hit.toi), eye.1 + f64::from(dir[1]) * f64::from(hit.toi), eye.2 + f64::from(dir[2]) * f64::from(hit.toi));
            let snd = block_sounds(&info.name).1.replace(".break", ".hit");
            world.play_custom_sound(&snd, SoundCategory::Blocks, at, 0.7, 0.7 + 0.2 * count as f32);
        }
    }
}

fn break_part(wname: &str, body: BodyId, part: u32, player: &pumpkin_plugin_api::Player) {
    let Some(vis) = STATE.with(|s| s.borrow().bodies.get(&(wname.to_string(), body)).cloned()) else { return };
    // A chest-like block keeps its contents: they travel inside the item it breaks into.
    let Some((state, nbt, offset, inv)) = vis.parts.borrow().iter().find(|p| p.id == part).map(|p| { snapshot(p); (p.state, p.nbt.clone(), p.offset, p.inv.borrow().clone()) }) else { return };
    let at = STATE.with(|s| s.borrow().engines.get(wname).and_then(|e| e.body_state(body))).map(|(pos, rot, _)| part_center(pos, rot, offset));

    // Survival: the item goes straight into the inventory, so refuse to break when there is no room
    // instead of destroying the block. A block with block-entity data needs a free slot (it never stacks).
    let item = if player.get_gamemode() == GameMode::Creative && inv.is_empty() { None } else { world::block_state_to_info(state).map(|i| i.name) };
    if let Some(name) = &item {
        let room = if nbt.is_some() || !inv.is_empty() { (0..player.get_inventory().as_inventory().get_size().min(36)).any(|i| player.get_inventory().as_inventory().get_item(i).is_none()) } else { can_give(player, name) };
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

    if let Some(name) = item { give(player, &name, nbt.as_deref(), &inv); }
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
    /// State of the tiny xorshift generator behind `rand01`.
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
pub(super) struct ExplodeHandler;
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
pub(super) fn process_blasts(server: &Server) {
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

// ---------------------------------------------------------------------------
// placing
// ---------------------------------------------------------------------------

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
    let held_contents = with_item_meta(held_inv(&stack), &stack, state);
    // The saved contents are authoritative; the item's block-entity Items would go stale once edited.
    let held_nbt = if held_contents.is_empty() { held_nbt } else { None };

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
        Some((pid, bs, is_full_block(&info, state)))
    });
    let Some((pid, (pos, rot, _), solid)) = part else { return };

    let center = part_center(pos, rot, off);
    match spawn_part(world, state, off, pid, center, solid, held_nbt) {
        Ok(p) => {
            *p.inv.borrow_mut() = held_contents;
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
// minecart contraptions: blocks riding on any ordinary minecart
// ---------------------------------------------------------------------------
//
// Any `minecraft:minecart` entity accepts blocks. Right-click it (or a block already on it) with a block
// in hand to add one; punch a block 3 times to take it off. The blocks are plain block displays (plus
// the usual client-side collision shulkers) that follow the cart: no physics while attached. They are
// saved in the minecart entity itself (block-entity NBT and chest contents included), and breaking the
// cart drops a minecart *item* that carries them (custom-data + enchantment glint); placing that item on
// a rail, or dispensing it onto one, brings the whole build back.
//
// Grid: cell (0,0,0) is the cart itself; its top is where layer 1 starts. Blocks may only be placed on
// layers >= 1 (they may overhang the cart sideways), never beside or under the cart.
//
// Rotation: Pumpkin does not keep a minecart's yaw up to date, so the heading is derived from the cart's
// own movement and snapped to quarter turns (so the collision shulkers stay exact). With `CART_FOLD` the
// house only turns when the cart changes *axis* (a curve), not when it simply reverses.

/// Fold the heading onto its axis: reversing along the same line does not spin the house.
const CART_FOLD: bool = true;
/// Key of the saved block list on the minecart entity (under `TAG_NS`), and of its saved quarter turn.
const CART_KEY: &str = "cart";
const CART_QUARTER_KEY: &str = "cartk";
/// Marks the saved block list (per block: offset, state, block-entity NBT, chest contents).
const CART_MAGIC: &[u8] = b"PMC2";
/// Height of a minecart's hitbox, and how far the grid origin sits above the entity's feet so that
/// layer 1 starts exactly on top of the cart.
const CART_H: f64 = 0.7;
const CART_ORIGIN_Y: f64 = CART_H - 0.5;

/// One saved block of a cart.
struct CartBlock {
    off: [i32; 3],
    state: u16,
    nbt: Option<Vec<u8>>,
    inv: Vec<u8>,
}

pub(super) struct Cart {
    pub(super) id: u32,
    pub(super) world: String,
    pub(super) ent: Entity,
    /// Visuals; shared type with physics bodies so `sync_colliders` serves carts too.
    pub(super) vis: Rc<BodyVis>,
    /// Next unused part id.
    next: StdCell<u32>,
    /// Where the visuals were last placed / where the cart was last seen.
    last: StdCell<[f64; 3]>,
    pos: StdCell<[f64; 3]>,
    /// Quarter turns of the blocks (0..=3) and the matching rotation.
    quarter: StdCell<u8>,
    rot: StdCell<[f32; 4]>,
    strikes: StdCell<u8>,
}

thread_local! {
    /// Carts whose destruction was announced (vehicle id, attacker id); handled in `cart_tick`.
    static CART_BREAKS: RefCell<Vec<(i32, Option<i32>)>> = const { RefCell::new(Vec::new()) };
}

fn cart_origin(p: [f64; 3]) -> [f64; 3] { [p[0], p[1] + CART_ORIGIN_Y, p[2]] }

/// Rotation about the vertical axis for `k` quarter turns (Minecraft yaw `k * 90`).
fn quarter_rot(k: u8) -> [f32; 4] {
    const H: f32 = std::f32::consts::FRAC_1_SQRT_2;
    match k & 3 { 0 => [0.0, 0.0, 0.0, 1.0], 1 => [0.0, -H, 0.0, H], 2 => [0.0, -1.0, 0.0, 0.0], _ => [0.0, H, 0.0, H] }
}

/// The quarter turn that matches the rail under a cart: 1 on an east-west rail, else 0.
fn rail_quarter(world: &World, pos: [f64; 3]) -> u8 {
    let st = world.get_block_state(BlockPos { x: pos[0].floor() as i32, y: (pos[1] + 0.1).floor() as i32, z: pos[2].floor() as i32 });
    if norm(&st.block_name).ends_with("rail") && st.properties.iter().any(|(k, v)| k == "shape" && v.ends_with("east_west")) { 1 } else { 0 }
}

// --- saved formats ---------------------------------------------------------

fn encode_blocks(parts: &[PartVis]) -> Vec<u8> {
    let mut out = CART_MAGIC.to_vec();
    for p in parts {
        for o in p.offset { out.extend_from_slice(&(o as i16).to_le_bytes()); }
        out.extend_from_slice(&p.state.to_le_bytes());
        let nbt = p.nbt.as_deref().unwrap_or(&[]);
        out.extend_from_slice(&(nbt.len() as u32).to_le_bytes());
        out.extend_from_slice(nbt);
        let inv = p.inv.borrow();
        out.extend_from_slice(&(inv.len() as u32).to_le_bytes());
        out.extend_from_slice(&inv);
    }
    out
}

fn take<'a>(r: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    if r.len() < n { return None; }
    let (a, b) = r.split_at(n);
    *r = b;
    Some(a)
}
fn take_u32(r: &mut &[u8]) -> Option<usize> { take(r, 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as usize) }

/// Damaged data ends the parse; what was read before is kept.
fn decode_blocks(b: &[u8]) -> Vec<CartBlock> {
    let mut out = Vec::new();
    let Some(mut r) = b.strip_prefix(CART_MAGIC) else { return out };
    while out.len() < MAX_PARTS {
        let Some(h) = take(&mut r, 8) else { break };
        let off = [i32::from(i16::from_le_bytes([h[0], h[1]])), i32::from(i16::from_le_bytes([h[2], h[3]])), i32::from(i16::from_le_bytes([h[4], h[5]]))];
        let state = u16::from_le_bytes([h[6], h[7]]);
        let Some(nbt) = take_u32(&mut r).and_then(|n| take(&mut r, n)).map(<[u8]>::to_vec) else { break };
        let Some(inv) = take_u32(&mut r).and_then(|n| take(&mut r, n)).map(<[u8]>::to_vec) else { break };
        out.push(CartBlock { off, state, nbt: if nbt.is_empty() { None } else { Some(nbt) }, inv });
    }
    out
}

/// A loaded cart is handed out as an allay spawn egg dressed up as a minecart. Pumpkin spawns the entity its
/// `entity_data` names and loads that data into it, so placing it, dispensing it and `/give` all work without
/// any event hooks; `convert_egg_cart` then swaps the freshly spawned allay for a minecart carrying the blocks.
const CART_EGG: &str = "minecraft:zombie_spawn_egg";
/// Entity type id of an allay, used only if the default egg doesn't reveal it (id in Pumpkin's entities.json).
const ALLAY_TYPE_ID_FALLBACK: u32 = 154; // zombie (name kept from the allay version)

/// A VarInt (ids above 127 take two bytes, so a lone byte would be a broken value).
fn varint_bytes(mut n: u32) -> Vec<u8> {
    let mut v = Vec::new();
    loop {
        let b = (n & 0x7f) as u8;
        n >>= 7;
        if n == 0 { v.push(b); return v; }
        v.push(b | 0x80);
    }
}
/// The leading VarInt of a component value.
fn varint_prefix(v: &[u8]) -> Vec<u8> {
    let n = v.iter().position(|b| b & 0x80 == 0).map_or(v.len(), |i| i + 1);
    v[..n].to_vec()
}
/// Starts a named NBT tag (type byte, then the name).
fn nbt_name(v: &mut Vec<u8>, tag: u8, name: &str) {
    v.push(tag);
    v.extend_from_slice(&(name.len() as u16).to_be_bytes());
    v.extend_from_slice(name.as_bytes());
}
/// `entity_data` for the egg: the allay's type id (read off the default egg) and the NBT
/// `{data: {<TAG_NS>: {cart: [B; ...]}}}`, which is where an entity's custom data lives.
fn egg_entity_data(blocks: &[u8]) -> Vec<u8> {
    let mut v = ItemStack::of(CART_EGG, 1).get_components().into_iter()
        .find(|c| is_comp(c.component, "entity-data")).map(|c| varint_prefix(&c.value))
        .filter(|p| !p.is_empty()).unwrap_or_else(|| varint_bytes(ALLAY_TYPE_ID_FALLBACK));
    v.push(10); // unnamed root compound
    // The zombie only exists for a moment before it becomes the cart: no AI, gravity, noise or damage, not a
    // baby.
    for flag in ["Invulnerable", "NoAI", "NoGravity", "Silent", "Invisible"] {
        nbt_name(&mut v, 1, flag);
        v.push(1);
    }
    nbt_name(&mut v, 1, "IsBaby");
    v.push(0);
    nbt_name(&mut v, 10, "data");
    nbt_name(&mut v, 10, TAG_NS);
    nbt_name(&mut v, 7, CART_KEY);
    v.extend_from_slice(&(blocks.len() as i32).to_be_bytes());
    v.extend_from_slice(blocks);
    v.extend_from_slice(&[0, 0, 0]); // end of TAG_NS, data, root
    v
}

/// A minecart item; with blocks it is the egg described at `CART_EGG`, looking like a minecart and glinting.
fn cart_item(blocks: &[u8]) -> ItemStack {
    if blocks.len() <= CART_MAGIC.len() { return ItemStack::of("minecraft:minecart", 1); }
    let s = ItemStack::of(CART_EGG, 1);
    if let Some(c) = comp_by_name("entity-data") { s.set_component(c, &egg_entity_data(blocks)); }
    if let Some(c) = comp_by_name("item-model") {
        let id = "minecraft:minecart";
        let mut v = vec![id.len() as u8];
        v.extend_from_slice(id.as_bytes());
        s.set_component(c, &v);
    }
    if let Some(c) = comp_by_name("item-name") {
        let text = "Minecart with blocks";
        let mut v = Vec::new();
        v.push(10); // unnamed compound {text: "..."}
        nbt_name(&mut v, 8, "text");
        v.extend_from_slice(&(text.len() as u16).to_be_bytes());
        v.extend_from_slice(text.as_bytes());
        v.push(0);
        s.set_component(c, &v);
    }
    if let Some(c) = comp_by_name("enchantment-glint-override") { s.set_component(c, &vec![1u8]); }
    s
}
/// The saved blocks an item carries: in the egg's entity-data.
fn item_blocks(stack: &ItemStack) -> Option<Vec<u8>> {
    stack.get_components().iter().find(|c| is_comp(c.component, "entity-data")).and_then(|c| custom_data_get(CART_KEY.as_bytes(), &c.value))
}

// --- building, saving, registering -----------------------------------------

fn build_parts(world: &World, pos: [f64; 3], rot: [f32; 4], blocks: &[CartBlock], first_id: u32) -> Vec<PartVis> {
    let o = cart_origin(pos);
    let mut view = View::new(world);
    let mut parts = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        let solid = is_full_block(&view.describe(b.state), b.state);
        if let Ok(p) = spawn_part(world, b.state, b.off, first_id + i as u32, part_center(o, rot, b.off), solid, b.nbt.clone()) {
            *p.inv.borrow_mut() = b.inv.clone();
            place_part_entities(&p, o, rot, true);
            parts.push(p);
        }
    }
    parts
}

/// Reads any open chest GUIs back into the parts' saved contents.
fn snap_all(c: &Cart) {
    if let Ok(parts) = c.vis.parts.try_borrow() { for p in parts.iter() { snapshot(p); } }
}

/// Saves the cart's blocks into the minecart entity (or clears them when it has none).
pub(super) fn save_cart(c: &Cart) {
    let Ok(parts) = c.vis.parts.try_borrow() else { return };
    if parts.is_empty() { c.ent.remove_custom_data(TAG_NS, CART_KEY); }
    else { c.ent.set_byte_array(TAG_NS, CART_KEY, encode_blocks(&parts).iter().map(|b| *b as i8).collect()); }
}

fn register_cart(world: &World, ent: Entity, blocks: &[CartBlock]) -> Option<Rc<Cart>> {
    let id = ent.get_id();
    if let Some(c) = STATE.with(|s| s.borrow().carts.get(&id).cloned()) { return Some(c); }
    let p = ent.get_position();
    let pos = [p.0, p.1, p.2];
    // A saved cart keeps its turn; a new one starts aligned with the rail it sits on.
    let saved = ent.get_int(TAG_NS, CART_QUARTER_KEY).map(|v| (v & 3) as u8);
    let k = saved.unwrap_or_else(|| if CART_FOLD { rail_quarter(world, pos) } else { 0 });
    if saved.is_none() { ent.set_int(TAG_NS, CART_QUARTER_KEY, i32::from(k)); }
    let rot = quarter_rot(k);
    let parts = build_parts(world, pos, rot, blocks, 0);
    let cart = Rc::new(Cart {
        id, world: world.get_name(), ent, next: StdCell::new(blocks.len() as u32),
        vis: Rc::new(BodyVis::new(parts, rot)), last: StdCell::new(pos), pos: StdCell::new(pos),
        quarter: StdCell::new(k), rot: StdCell::new(rot), strikes: StdCell::new(0),
    });
    STATE.with(|s| { s.borrow_mut().carts.insert(id, cart.clone()); });
    save_cart(&cart);
    Some(cart)
}

/// Called by the recovery scan for every minecart: picks up carts that have saved blocks.
pub(super) fn cart_adopt(world: &World, e: Entity) {
    if STATE.with(|s| s.borrow().carts.contains_key(&e.get_id())) { return; }
    let Some(saved) = e.get_byte_array(TAG_NS, CART_KEY) else { return };
    let bytes: Vec<u8> = saved.iter().map(|b| *b as u8).collect();
    let blocks = decode_blocks(&bytes);
    if blocks.is_empty() { return; }
    register_cart(world, e, &blocks);
}

/// Rebuilds the visuals of carts whose displays were lost to a chunk unload (same two-strike rule as bodies).
pub(super) fn cart_repair(server: &Server, loaded: &HashSet<u32>) -> bool {
    let carts: Vec<Rc<Cart>> = STATE.with(|s| s.borrow().carts.values().cloned().collect());
    let mut again = false;
    for c in carts {
        let Some(world) = server.get_world_by_name(&c.world) else { continue };
        let p = c.pos.get();
        let missing = c.vis.parts.try_borrow().is_ok_and(|parts| parts.iter().any(|pt| {
            !loaded.contains(&pt.did) && world.get_chunk((p[0].floor() as i32) >> 4, (p[2].floor() as i32) >> 4).is_some()
        }));
        if !missing { c.strikes.set(0); continue; }
        c.strikes.set(c.strikes.get().saturating_add(1));
        if c.strikes.get() < 2 { again = true; continue; }
        c.strikes.set(0);
        snap_all(&c);
        let old: Vec<PartVis> = c.vis.parts.borrow_mut().drain(..).collect();
        let blocks: Vec<CartBlock> = old.iter().map(|pt| CartBlock { off: pt.offset, state: pt.state, nbt: pt.nbt.clone(), inv: pt.inv.borrow().clone() }).collect();
        for pt in old { kill_part(pt); }
        let fresh = build_parts(&world, p, c.rot.get(), &blocks, 0);
        c.next.set(fresh.len() as u32);
        *c.vis.parts.borrow_mut() = fresh;
        c.last.set(p);
    }
    again
}

/// Removes a cart's visuals and forgets it (the saved blocks stay on the minecart entity).
fn retire_cart(c: &Rc<Cart>) {
    let parts: Vec<PartVis> = c.vis.parts.borrow_mut().drain(..).collect();
    for p in parts { kill_part(p); }
    STATE.with(|s| { s.borrow_mut().carts.remove(&c.id); });
}

/// How far from a cart a player can be and still receive its item when nobody is named.
const CART_GIVE_RADIUS: f64 = 16.0;

/// The v0.1 plugin API has no `World::drop_item`, so a cart's item goes straight into a player's inventory
/// instead: `prefer` (the player who broke it) first, then anyone in the same world within
/// `CART_GIVE_RADIUS`, nearest first. Returns false when nobody has a free slot (minecarts don't stack).
fn give_cart_item(server: &Server, c: &Cart, at: [f64; 3], prefer: Option<i32>) -> bool {
    snap_all(c);
    let bytes = c.vis.parts.try_borrow().map(|p| encode_blocks(&p)).unwrap_or_default();
    let mut near: Vec<(bool, f64, pumpkin_plugin_api::Player)> = server.get_all_players().into_iter().filter_map(|p| {
        if p.get_world().get_name() != c.world { return None; }
        let pos = p.as_entity().get_position();
        let d2 = (pos.0 - at[0]).powi(2) + (pos.1 - at[1]).powi(2) + (pos.2 - at[2]).powi(2);
        let chosen = prefer == Some(p.as_entity().get_id() as i32);
        (chosen || d2 <= CART_GIVE_RADIUS * CART_GIVE_RADIUS).then_some((!chosen, d2, p))
    }).collect();
    near.sort_by(|a, b| (a.0, a.1).partial_cmp(&(b.0, b.1)).unwrap_or(std::cmp::Ordering::Equal));
    for (_, _, p) in near {
        let inv = p.get_inventory().as_inventory();
        if let Some(slot) = (0..inv.get_size().min(36)).find(|i| inv.get_item(*i).is_none()) {
            inv.set_item(slot, Some(cart_item(&bytes)));
            return true;
        }
    }
    false
}

/// The v0.1 API also has no `World::get_entity_by_id`; look the entity up among `anchor`'s neighbours
/// first (cheap), then fall back to scanning the whole world.
fn find_entity(world: &World, anchor: &Entity, id: i32) -> Option<Entity> {
    anchor.get_nearby_entities(8.0, 8.0, 8.0).into_iter().find(|e| e.get_id() as i32 == id)
        .or_else(|| world.get_entities().into_iter().find(|e| e.get_id() as i32 == id))
}

/// Writes every open chest's contents (and each cart's blocks) into the saved data. Used on unload and
/// by the periodic storage sync; `force` saves even without changes.
pub(super) fn save_carts(force: bool) {
    let carts: Vec<Rc<Cart>> = STATE.with(|s| s.borrow().carts.values().cloned().collect());
    for c in carts {
        let before: Vec<Vec<u8>> = c.vis.parts.try_borrow().map(|p| p.iter().map(|x| x.inv.borrow().clone()).collect()).unwrap_or_default();
        snap_all(&c);
        let changed = c.vis.parts.try_borrow().is_ok_and(|p| p.iter().zip(&before).any(|(x, b)| *x.inv.borrow() != *b));
        if force || changed { save_cart(&c); }
    }
}

/// Per-tick: handles broken carts, and moves every cart's blocks along with it (turning them with it).
pub(super) fn cart_tick(server: &Server) {
    convert_egg_carts(server);
    let breaks = CART_BREAKS.with(|b| std::mem::take(&mut *b.borrow_mut()));
    for (eid, attacker) in breaks {
        let Some(c) = STATE.with(|s| s.borrow().carts.get(&(eid as u32)).cloned()) else { continue };
        let creative = attacker.is_some_and(|a| server.get_all_players().iter().any(|p| p.as_entity().get_id() as i32 == a && p.get_gamemode() == GameMode::Creative));
        let has_blocks = c.vis.parts.try_borrow().is_ok_and(|p| !p.is_empty());
        let at = c.pos.get();
        // Nobody with room for the item: the vanilla break was already cancelled, so just leave the cart
        // standing (blocks and all) rather than lose the build.
        if (!creative || has_blocks) && !give_cart_item(server, &c, at, attacker) { continue; }
        c.ent.remove();
        retire_cart(&c);
    }
    let carts: Vec<Rc<Cart>> = STATE.with(|s| s.borrow().carts.values().cloned().collect());
    // No `get_entity_by_id` in the v0.1 API: every 10 ticks, list each world's entity ids once and check
    // the carts against that.
    let check_alive = now_tick() % 10 == 0;
    let mut live: HashMap<String, HashSet<u32>> = HashMap::new();
    for c in carts {
        let Some(world) = server.get_world_by_name(&c.world) else { continue };
        if check_alive {
            let ids = live.entry(c.world.clone()).or_insert_with(|| world.get_entities().iter().map(|e| e.get_id()).collect());
            if !ids.contains(&c.id) {
                // Gone. With its chunk unloaded that is just a save (the blocks stay in the entity and come back
                // with it); with the chunk loaded the cart was killed some other way, so its blocks go to the
                // nearest player as an item (if there is one with room).
                let p = c.pos.get();
                if world.get_chunk((p[0].floor() as i32) >> 4, (p[2].floor() as i32) >> 4).is_some() { give_cart_item(server, &c, p, None); }
                retire_cart(&c);
                continue;
            }
        }
        let prev = c.pos.get();
        let pp = c.ent.get_position();
        let p = [pp.0, pp.1, pp.2];
        c.pos.set(p);

        // Heading from movement: the dominant axis decides (with some hysteresis for diagonals).
        let (dx, dz) = (p[0] - prev[0], p[2] - prev[2]);
        let mut k = c.quarter.get();
        if dx * dx + dz * dz > 4e-4 {
            let heading = if dx.abs() > dz.abs() * 1.3 { if dx > 0.0 { 3 } else { 1 } }
                else if dz.abs() > dx.abs() * 1.3 { if dz > 0.0 { 0 } else { 2 } }
                else { k };
            // Folded: only a change of axis (a curve) turns the blocks, and then to face the new direction of
            // travel (a right turn turns them right, a left turn left); reversing along the line keeps them.
            k = if CART_FOLD { if (heading & 1) != (k & 1) { heading } else { k } } else { heading };
        }
        let turned = k != c.quarter.get();
        if turned {
            c.quarter.set(k);
            c.rot.set(quarter_rot(k));
            c.ent.set_int(TAG_NS, CART_QUARTER_KEY, i32::from(k));
        }

        let lp = c.last.get();
        let moved = (p[0] - lp[0]).abs() > POS_EPS || (p[1] - lp[1]).abs() > POS_EPS || (p[2] - lp[2]).abs() > POS_EPS;
        if !moved && !turned { continue; }
        c.last.set(p);
        let (o, rot) = (cart_origin(p), c.rot.get());
        if let Ok(parts) = c.vis.parts.try_borrow() {
            for part in parts.iter() { place_part_entities(part, o, rot, turned); }
        }
    }
}

// --- events -----------------------------------------------------------------

/// Cancels the vanilla destruction of a cart that carries blocks, so `cart_tick` can drop the right item.
pub(super) struct CartDestroyHandler;
impl EventHandler<VehicleDestroyEvent> for CartDestroyHandler {
    fn handle(&self, _server: Server, mut ev: VehicleDestroyEventData) -> VehicleDestroyEventData {
        // Never block on `STATE` here (see `ExplodeHandler`).
        let ours = STATE.with(|s| s.try_borrow().is_ok_and(|st| st.carts.contains_key(&(ev.vehicle_id as u32))));
        if ours && !ev.cancelled {
            ev.cancelled = true;
            CART_BREAKS.with(|b| b.borrow_mut().push((ev.vehicle_id, ev.attacker_id)));
        }
        ev
    }
}

/// Where the cart goes: on the rail at or just under the spawned allay, else where the allay stood.
fn egg_cart_spot(world: &World, p: (f64, f64, f64)) -> (f64, f64, f64) {
    let (x, z) = (p.0.floor() as i32, p.2.floor() as i32);
    let y = (p.1 + 0.1).floor() as i32;
    for dy in [0, -1] {
        let s = world.get_block_state(BlockPos { x, y: y + dy, z });
        if norm(&s.block_name).ends_with("rail") {
            let asc = s.properties.iter().any(|(k, v)| k == "shape" && v.starts_with("ascending"));
            return (f64::from(x) + 0.5, f64::from(y + dy) + 0.0625 + if asc { 0.5 } else { 0.0 }, f64::from(z) + 0.5);
        }
    }
    p
}

/// An allay carrying saved cart blocks (it came out of a cart egg) is replaced by a minecart with them.
/// False for an ordinary allay.
fn convert_egg_cart(world: &World, e: &Entity) -> bool {
    let Some(saved) = e.get_byte_array(TAG_NS, CART_KEY) else { return false };
    let blocks = decode_blocks(&saved.iter().map(|b| *b as u8).collect::<Vec<u8>>());
    if blocks.is_empty() { return false; }
    let at = egg_cart_spot(world, e.get_position());
    e.remove();
    let cart = world.spawn_entity(EntityType::Minecart, at);
    register_cart(world, cart, &blocks);
    true
}

/// Called by the recovery scan for every allay (catches one the spawn event missed).
pub(super) fn egg_cart_adopt(world: &World, e: &Entity) { convert_egg_cart(world, e); }

/// How often (ticks) the entities are checked for a cart egg that something other than a player click spawned
/// (dispenser, `/summon`, `/give` + other mods). Players placing the egg never wait for this.
const EGG_SCAN_EVERY_TICKS: u64 = 4;

/// Finds cart eggs spawned without a click (dispensers). There is deliberately no EntitySpawnEvent handler (it
/// deadlocks the server, see `Physics::on_load`), so this walks the entities every few ticks while players are
/// online. The placeholder zombie is invisible and has no AI, so the short wait is not noticeable.
fn convert_egg_carts(server: &Server) {
    let now = STATE.with(|s| s.borrow().ticks);
    // Offset 3 keeps clear of the recovery scan (ticks 1 and 2 of every 60).
    if now % EGG_SCAN_EVERY_TICKS != 3 { return; }
    if server.get_all_players().is_empty() { return; }
    for world in server.get_all_worlds() {
        for e in world.get_entities() {
            if matches!(e.get_type(), EntityType::Zombie) { convert_egg_cart(&world, &e); }
        }
    }
}

// --- interaction ------------------------------------------------------------

fn is_cart_collider(id: i32) -> bool {
    STATE.with(|s| s.try_borrow().is_ok_and(|st| st.carts.values().any(|c| {
        c.vis.parts.try_borrow().is_ok_and(|ps| ps.iter().any(|p| p.col.as_ref().is_some_and(|col| col.eid == id)))
    })))
}

/// The cart and block cell whose client-side collider uses entity id `id`.
fn cart_part_by_collider(id: i32) -> Option<(Rc<Cart>, [i32; 3])> {
    STATE.with(|s| {
        let st = s.try_borrow().ok()?;
        st.carts.values().find_map(|c| {
            let ps = c.vis.parts.try_borrow().ok()?;
            ps.iter().find(|p| p.col.as_ref().is_some_and(|col| col.eid == id)).map(|p| (c.clone(), p.offset))
        })
    })
}

/// Ray vs axis-aligned box; returns the entry distance and the face normal hit.
fn ray_box(eye: [f64; 3], dir: [f32; 3], lo: [f64; 3], hi: [f64; 3]) -> Option<(f64, [i32; 3])> {
    let (mut tmin, mut tmax) = (0.0f64, f64::MAX);
    let mut n = [0i32; 3];
    for i in 0..3 {
        let d = f64::from(dir[i]);
        if d.abs() < 1e-9 {
            if eye[i] < lo[i] || eye[i] > hi[i] { return None; }
            continue;
        }
        let (mut t1, mut t2) = ((lo[i] - eye[i]) / d, (hi[i] - eye[i]) / d);
        let mut s = -1;
        if t1 > t2 { std::mem::swap(&mut t1, &mut t2); s = 1; }
        if t1 > tmin { tmin = t1; n = [0; 3]; n[i] = s; }
        tmax = tmax.min(t2);
        if tmin > tmax { return None; }
    }
    Some((tmin, n))
}

/// `cell` and `normal` are in the cart's own (unturned) grid; `is_cart` means the cart itself was hit.
struct CartHit { cart: Rc<Cart>, cell: [i32; 3], normal: [i32; 3], is_cart: bool }

/// Nearest cart (or block on a cart) under the crosshair, unless a real block is in front of it.
fn cart_hit(player: &pumpkin_plugin_api::Player, reach: f32) -> Option<CartHit> {
    let world = player.get_world();
    let wname = world.get_name();
    let carts: Vec<Rc<Cart>> = STATE.with(|s| s.try_borrow().map(|st| st.carts.values().filter(|c| c.world == wname).cloned().collect()).unwrap_or_default());
    if carts.is_empty() { return None; }
    let e = player.as_entity();
    let (eye, dir) = (e.get_eye_position(), look_dir(&e));
    let eye3 = [eye.0, eye.1, eye.2];
    let mut best: Option<(f64, CartHit)> = None;
    for c in carts {
        let p = c.pos.get();
        if dist_sq((p[0], p[1], p[2]), eye) > (f64::from(reach) + 24.0).powi(2) { continue; }
        let (o, rot) = (cart_origin(p), c.rot.get());
        let mut cands: Vec<([f64; 3], [f64; 3], [i32; 3], bool)> = vec![([p[0] - 0.49, p[1], p[2] - 0.49], [p[0] + 0.49, p[1] + CART_H, p[2] + 0.49], [0, 0, 0], true)];
        if let Ok(parts) = c.vis.parts.try_borrow() {
            for pt in parts.iter() {
                let m = part_center(o, rot, pt.offset);
                // A hair larger than the block: the client picks entities with some slack and the cart may have moved a tick.
                cands.push(([m.0 - 0.52, m.1 - 0.52, m.2 - 0.52], [m.0 + 0.52, m.1 + 0.52, m.2 + 0.52], pt.offset, false));
            }
        }
        let conj = [-rot[0], -rot[1], -rot[2], rot[3]];
        for (lo, hi, cell, is_cart) in cands {
            let Some((t, n)) = ray_box(eye3, dir, lo, hi) else { continue };
            if n == [0, 0, 0] || t > f64::from(reach) || best.as_ref().is_some_and(|b| b.0 <= t) { continue; }
            // The face normal is in world space; the grid is in the cart's turned frame.
            let l = rotate(conj, [n[0] as f32, n[1] as f32, n[2] as f32]);
            let normal = [l[0].round() as i32, l[1].round() as i32, l[2].round() as i32];
            best = Some((t, CartHit { cart: c.clone(), cell, normal, is_cart }));
        }
    }
    let (t, hit) = best?;
    if shielded(&world, eye, dir, reach, t as f32) { None } else { Some(hit) }
}

fn held_block(player: &pumpkin_plugin_api::Player) -> Option<(ItemStack, u16)> {
    let stack = player.get_item_in_hand(Hand::Right)?;
    let state = world::get_block_by_name(&stack.get_registry_key())?.default_state_id;
    Some((stack, state))
}

fn place_on_cart(player: &pumpkin_plugin_api::Player, cart: &Rc<Cart>, target: [i32; 3]) {
    if target[1] < 1 { return; }
    let Some((stack, state)) = held_block(player) else { return };
    {
        let Ok(parts) = cart.vis.parts.try_borrow() else { return };
        if parts.len() >= MAX_PARTS || parts.iter().any(|p| p.offset == target) { return; }
    }
    // An item that carries block entity data (a chest broken out of a build) keeps it.
    let held_nbt: Option<Vec<u8>> = stack
        .get_components()
        .into_iter()
        .find(|c| is_comp(c.component, BLOCK_ENTITY_DATA))
        .and_then(|c| c.value.split_first().map(|(_, rest)| rest.to_vec()))
        .filter(|v| !v.is_empty());
    let held_contents = with_item_meta(held_inv(&stack), &stack, state);
    // The saved contents are authoritative; the item's block-entity Items would go stale once edited.
    let held_nbt = if held_contents.is_empty() { held_nbt } else { None };
    let world = player.get_world();
    let solid = is_full_block(&View::new(&world).describe(state), state);
    let id = cart.next.get();
    cart.next.set(id + 1);
    let (o, rot) = (cart_origin(cart.pos.get()), cart.rot.get());
    let center = part_center(o, rot, target);
    let Ok(p) = spawn_part(&world, state, target, id, center, solid, held_nbt) else { return };
    *p.inv.borrow_mut() = held_contents;
    place_part_entities(&p, o, rot, true);
    cart.vis.parts.borrow_mut().push(p);
    save_cart(cart);
    if let Some(info) = world::block_state_to_info(state) {
        world.play_custom_sound(block_sounds(&info.name).0, SoundCategory::Blocks, center, 1.0, 0.8);
    }
    if player.get_gamemode() != GameMode::Creative {
        let n = stack.get_count();
        if n > 1 { stack.set_count(n - 1); player.set_item_in_hand(Hand::Right, Some(stack)); }
        else { player.set_item_in_hand(Hand::Right, None); }
    }
}

/// Opens a chest-like block that rides on a cart (same storage as on physics bodies). True when it did.
fn cart_try_open(player: &pumpkin_plugin_api::Player, hit: &CartHit, now: u64) -> bool {
    let parts = hit.cart.vis.parts.borrow();
    let Some(p) = parts.iter().find(|p| p.offset == hit.cell) else { return false };
    let Some(kind) = world::block_state_to_info(p.state).and_then(|i| interaction(&i.name)) else { return false };
    open_part(player, p, kind, now)
}

/// Right-click on a cart or on a block that is already on one: opens a chest-like block, or places the
/// held block (`via`: the minecart entity that was clicked, if any). True when the click was ours.
fn cart_use(player: &pumpkin_plugin_api::Player, via: Option<Entity>) -> bool {
    let pid = player.as_entity().get_id();
    let now = now_tick();
    let hit = cart_hit(player, REACH);
    if let Some(h) = &hit {
        if !h.is_cart && !is_sneaking(player) && cart_try_open(player, h, now) { return true; }
    }
    if held_block(player).is_none() { return false; }
    // Holding the button repeats the click; place one block per few ticks.
    if STATE.with(|s| s.borrow().last_action.get(&pid).is_some_and(|t| now.saturating_sub(*t) < 3)) { return true; }
    STATE.with(|s| { s.borrow_mut().last_action.insert(pid, now); });
    let (cart, target) = match hit {
        Some(h) if h.is_cart => (h.cart, [0, 1, 0]),
        Some(h) => (h.cart, [h.cell[0] + h.normal[0], h.cell[1] + h.normal[1], h.cell[2] + h.normal[2]]),
        None => {
            let Some(ent) = via else { return false };
            let Some(c) = register_cart(&player.get_world(), ent, &[]) else { return false };
            (c, [0, 1, 0])
        }
    };
    place_on_cart(player, &cart, target);
    true
}

/// A right-click on a minecart entity. Holding a block turns it into a placement; otherwise the click is
/// left to the game (sit in it).
fn cart_entity_click(player: &pumpkin_plugin_api::Player, entity_id: i32) -> bool {
    if held_block(player).is_none() { return false; }
    let Some(ent) = find_entity(&player.get_world(), &player.as_entity(), entity_id) else { return false };
    if !matches!(ent.get_type(), EntityType::Minecart) { return false; }
    cart_use(player, Some(ent))
}

/// Right-click with a minecart item that carries blocks (the cart egg): puts the cart and its blocks back, on the
/// rail that was clicked or on top of the clicked block.
fn try_place_cart(player: &pumpkin_plugin_api::Player, pos: BlockPos, on_rail: bool) -> bool {
    let Some(stack) = player.get_item_in_hand(Hand::Right) else { return false };
    let registry_key = stack.get_registry_key();
    let key = norm(&registry_key);
    if key != "minecart" && key != norm(CART_EGG) { return false; }
    let Some(bytes) = item_blocks(&stack) else { return false };
    let blocks = decode_blocks(&bytes);
    if blocks.is_empty() { return false; }
    let world = player.get_world();
    let top = if on_rail { f64::from(pos.y) + 0.0625 } else { f64::from(pos.y) + 1.0 };
    let at = egg_cart_spot(&world, (f64::from(pos.x) + 0.5, top, f64::from(pos.z) + 0.5));
    let ent = world.spawn_entity(EntityType::Minecart, at);
    register_cart(&world, ent, &blocks);
    if player.get_gamemode() != GameMode::Creative { player.set_item_in_hand(Hand::Right, None); }
    true
}

/// Counts arm swings on a block that rides on a cart; the third one within a second takes it off.
/// True when the swing was aimed at such a block.
fn cart_punch(player: &pumpkin_plugin_api::Player) -> bool {
    if STATE.with(|s| s.try_borrow().map_or(true, |st| st.carts.is_empty())) { return false; }
    let pid = player.as_entity().get_id();
    let Some(h) = cart_hit(player, BREAK_REACH).filter(|h| !h.is_cart) else {
        STATE.with(|s| { s.borrow_mut().cart_punches.remove(&pid); });
        return false;
    };
    let count = STATE.with(|s| {
        let mut st = s.borrow_mut();
        let now = st.ticks;
        let e = st.cart_punches.entry(pid).or_insert((h.cart.id, h.cell, Vec::new()));
        if e.0 != h.cart.id || e.1 != h.cell { *e = (h.cart.id, h.cell, Vec::new()); }
        e.2.retain(|t| now.saturating_sub(*t) < PUNCH_WINDOW_TICKS);
        e.2.push(now);
        e.2.len()
    });
    if count >= PUNCHES_TO_BREAK {
        STATE.with(|s| { s.borrow_mut().cart_punches.remove(&pid); });
        // Do NOT break the block here: this runs inside the PlayerAnimation event handler, and breaking
        // spawns entities (loose blocks becoming bodies). Spawning fires EntitySpawnEvent, which the server
        // dispatches back into this very plugin while it is still busy with the swing -> deadlock that
        // freezes the whole server. Queue it and let `process_cart_breaks` do it from the tick instead.
        PUNCH_CART_BREAKS.with(|q| { let mut q = q.borrow_mut(); if q.len() < 64 { q.push((pid, h.cart.id, h.cell)); } });
    }
    true
}

thread_local! {
    /// (player entity id, cart id, cell) of cart blocks whose third punch landed; see `cart_punch`.
    static PUNCH_CART_BREAKS: RefCell<Vec<(u32, u32, [i32; 3])>> = const { RefCell::new(Vec::new()) };
}

/// Runs the cart block breaks queued by `cart_punch`, from the tick (outside any event handler).
pub(super) fn process_cart_breaks(server: &Server) {
    let pending = PUNCH_CART_BREAKS.with(|q| std::mem::take(&mut *q.borrow_mut()));
    if pending.is_empty() { return; }
    let players = server.get_all_players();
    for (pid, cart_id, cell) in pending {
        let Some(player) = players.iter().find(|p| p.as_entity().get_id() == pid) else { continue };
        let Some(cart) = STATE.with(|s| s.borrow().carts.get(&cart_id).cloned()) else { continue };
        break_cart_block(player, &cart, cell);
    }
}

/// Blocks that are no longer connected (face to face) to the cart fall off and become normal physics bodies.
fn drop_loose_blocks(world: &World, cart: &Rc<Cart>) {
    const DIRS: [[i32; 3]; 6] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    let loose: Vec<PartVis> = {
        let Ok(mut parts) = cart.vis.parts.try_borrow_mut() else { return };
        let cells: HashSet<[i32; 3]> = parts.iter().map(|p| p.offset).collect();
        // The cart itself is cell [0, 0, 0]; everything reachable from it stays on.
        let mut seen: HashSet<[i32; 3]> = HashSet::from([[0, 0, 0]]);
        let mut stack = vec![[0, 0, 0]];
        while let Some(c) = stack.pop() {
            for d in DIRS {
                let n = [c[0] + d[0], c[1] + d[1], c[2] + d[2]];
                if cells.contains(&n) && seen.insert(n) { stack.push(n); }
            }
        }
        let mut out = Vec::new();
        let mut i = 0;
        while i < parts.len() {
            if seen.contains(&parts[i].offset) { i += 1; } else { out.push(parts.remove(i)); }
        }
        out
    };
    if loose.is_empty() { return; }
    cart.vis.invalidate();
    let (o, rot) = (cart_origin(cart.pos.get()), cart.rot.get());
    // Split the loose blocks into connected groups; each becomes its own body.
    let mut left = loose;
    while let Some(first) = left.pop() {
        let mut group = vec![first];
        loop {
            let at = left.iter().position(|p| group.iter().any(|g| {
                let d = [(g.offset[0] - p.offset[0]).abs(), (g.offset[1] - p.offset[1]).abs(), (g.offset[2] - p.offset[2]).abs()];
                d[0] + d[1] + d[2] == 1
            }));
            match at { Some(i) => group.push(left.remove(i)), None => break }
        }
        let anchor = group[0].offset;
        let blocks: Vec<([i32; 3], u16)> = group.iter().map(|p| (p.offset, p.state)).collect();
        let nbts: Vec<Option<Vec<u8>>> = group.iter().map(|p| p.nbt.clone()).collect();
        let invs: Vec<Vec<u8>> = group.iter().map(|p| { snapshot(p); p.inv.borrow().clone() }).collect();
        let center = part_center(o, rot, anchor);
        for p in group { kill_part(p); }
        if let Ok(bid) = spawn_body_rotated(world, &blocks, anchor, Some([center.0, center.1, center.2]), Some(rot), &nbts) {
            if let Some(vis) = STATE.with(|s| s.borrow().bodies.get(&(world.get_name(), bid)).cloned()) {
                for (i, inv) in invs.iter().enumerate() {
                    if inv.is_empty() { continue; }
                    if let Some(p) = vis.parts.borrow().iter().find(|p| p.id == i as u32) {
                        *p.inv.borrow_mut() = inv.clone();
                        p.display.set_byte_array(TAG_NS, "inv", inv.iter().map(|b| *b as i8).collect());
                    }
                }
            }
        }
    }
}

fn break_cart_block(player: &pumpkin_plugin_api::Player, cart: &Rc<Cart>, cell: [i32; 3]) {
    let (state, nbt, saved_inv) = {
        let parts = cart.vis.parts.borrow();
        let Some(p) = parts.iter().find(|p| p.offset == cell) else { return };
        snapshot(p);
        (p.state, p.nbt.clone(), p.inv.borrow().clone())
    };
    let name = world::block_state_to_info(state).map(|i| i.name);
    let creative = player.get_gamemode() == GameMode::Creative;
    // A chest-like block keeps its contents (inside the item), so even creative players get that item.
    let hand_over = !creative || !saved_inv.is_empty();
    if hand_over {
        if let Some(n) = &name {
            // A block with block-entity data or contents never stacks, so it needs a free slot.
            let inv = player.get_inventory().as_inventory();
            let room = if nbt.is_some() || !saved_inv.is_empty() { (0..inv.get_size().min(36)).any(|i| inv.get_item(i).is_none()) } else { can_give(player, n) };
            if !room {
                player.send_system_message(TextComponent::text("Your inventory is full"), true);
                return;
            }
        }
    }
    let removed = {
        let mut parts = cart.vis.parts.borrow_mut();
        parts.iter().position(|p| p.offset == cell).map(|i| parts.remove(i))
    };
    let Some(p) = removed else { return };
    let at = part_center(cart_origin(cart.pos.get()), cart.rot.get(), p.offset);
    kill_part(p);
    drop_loose_blocks(&player.get_world(), cart);
    save_cart(cart);
    if let Some(n) = &name {
        player.get_world().play_custom_sound(block_sounds(n).1, SoundCategory::Blocks, at, 1.0, 0.8);
        if hand_over { give(player, n, nbt.as_deref(), &saved_inv); }
    }
}
