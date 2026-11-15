//! Plugin configuration: a plain `key = value` file (`config.properties`) in the plugin's data folder.
//! It is created with all defaults and comments on first start; edit it and restart the server (or reload the plugin).
//! Unknown keys and unparsable values are ignored (the default is used), so a typo never breaks the plugin.

use std::cell::Cell;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct PluginConfig {
    // --- tree felling ---
    /// Breaking a log of a natural tree turns the tree into falling physics blocks.
    pub trees_enabled: bool,
    /// Logs (connected) needed for a log structure to count as a tree.
    pub tree_min_logs: usize,
    /// Leaves of the same species touching the logs needed to count as a tree.
    pub tree_min_leaves: usize,
    /// Upper limit of connected logs; bigger blobs are treated as forests/builds and not felled.
    pub tree_max_logs: usize,
    /// The trunk must stand on dirt / grass / podzol / sand... (not on a built floor).
    pub tree_require_ground: bool,
    /// Max blocks (logs first, then leaves) of one felled tree.
    pub tree_max_blocks: usize,
    /// How many blocks away from the trunk leaves still come along.
    pub tree_leaf_reach: u32,
    /// Sideways push of the tree top (blocks/s) = base + per_tonne * mass in tonnes, capped at max.
    pub tree_push_base: f32,
    pub tree_push_per_tonne: f32,
    pub tree_push_max: f32,
    // --- leaves ---
    /// Leaves in physics bodies have no collision at all (they pass through everything).
    pub leaves_ghost: bool,
    /// Chance per second that a leaf block disappears.
    pub leaf_decay_chance_per_second: f32,
    /// A leaf that is inside a block disappears.
    pub leaf_vanish_inside_blocks: bool,
    /// How often (ticks) leaves are checked.
    pub leaf_check_interval_ticks: u64,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            trees_enabled: true,
            tree_min_logs: 3,
            tree_min_leaves: 4,
            tree_max_logs: 250,
            tree_require_ground: true,
            tree_max_blocks: 100,
            tree_leaf_reach: 4,
            tree_push_base: 1.0,
            tree_push_per_tonne: 0.8,
            tree_push_max: 6.0,
            leaves_ghost: true,
            leaf_decay_chance_per_second: 0.1,
            leaf_vanish_inside_blocks: true,
            leaf_check_interval_ticks: 5,
        }
    }
}

thread_local! { static CONFIG: Cell<PluginConfig> = Cell::new(PluginConfig::default()); }

/// The current configuration (a cheap copy).
pub fn cfg() -> PluginConfig { CONFIG.with(Cell::get) }

const HEADER: &str = "# Physics plugin configuration. Lines are `key = value`; `#` starts a comment.\n# Restart the server (or reload the plugin) after editing.\n";

fn describe(c: &PluginConfig) -> String {
    format!(
"{HEADER}
# --- Tree felling ---
# Breaking a log of a natural tree turns the tree into falling physics blocks.
trees_enabled = {}
# Connected logs needed for a log structure to count as a tree.
tree_min_logs = {}
# Leaves of the same species touching the logs needed to count as a tree (keeps log cabins safe).
tree_min_leaves = {}
# Log blobs bigger than this are not trees (forests, big builds).
tree_max_logs = {}
# The trunk must stand on dirt / grass / podzol / sand ... and not on a built floor.
tree_require_ground = {}
# Max blocks (logs first, then leaves) in one felled tree.
tree_max_blocks = {}
# How many blocks away from the trunk leaves still come along.
tree_leaf_reach = {}
# Sideways push of the tree top in blocks/s: base + per_tonne * (tree mass in tonnes), capped at max.
# Raise these if big trees don't fall over.
tree_push_base = {}
tree_push_per_tonne = {}
tree_push_max = {}

# --- Leaves ---
# Leaf blocks in physics bodies have no collision at all.
leaves_ghost = {}
# Chance per second that a leaf block disappears (0.1 = 1 in 10).
leaf_decay_chance_per_second = {}
# Leaves that are inside a block disappear.
leaf_vanish_inside_blocks = {}
# How often leaves are checked, in ticks (20 ticks = 1 second).
leaf_check_interval_ticks = {}
",
        c.trees_enabled, c.tree_min_logs, c.tree_min_leaves, c.tree_max_logs, c.tree_require_ground, c.tree_max_blocks,
        c.tree_leaf_reach, c.tree_push_base, c.tree_push_per_tonne, c.tree_push_max,
        c.leaves_ghost, c.leaf_decay_chance_per_second, c.leaf_vanish_inside_blocks, c.leaf_check_interval_ticks,
    )
}

fn parse(text: &str, mut c: PluginConfig) -> PluginConfig {
    fn p<T: std::str::FromStr>(v: &str, slot: &mut T) { if let Ok(x) = v.parse() { *slot = x; } }
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        let Some((k, v)) = line.split_once('=') else { continue };
        let (k, v) = (k.trim(), v.trim());
        match k {
            "trees_enabled" => p(v, &mut c.trees_enabled),
            "tree_min_logs" => p(v, &mut c.tree_min_logs),
            "tree_min_leaves" => p(v, &mut c.tree_min_leaves),
            "tree_max_logs" => p(v, &mut c.tree_max_logs),
            "tree_require_ground" => p(v, &mut c.tree_require_ground),
            "tree_max_blocks" => p(v, &mut c.tree_max_blocks),
            "tree_leaf_reach" => p(v, &mut c.tree_leaf_reach),
            "tree_push_base" => p(v, &mut c.tree_push_base),
            "tree_push_per_tonne" => p(v, &mut c.tree_push_per_tonne),
            "tree_push_max" => p(v, &mut c.tree_push_max),
            "leaves_ghost" => p(v, &mut c.leaves_ghost),
            "leaf_decay_chance_per_second" => p(v, &mut c.leaf_decay_chance_per_second),
            "leaf_vanish_inside_blocks" => p(v, &mut c.leaf_vanish_inside_blocks),
            "leaf_check_interval_ticks" => p(v, &mut c.leaf_check_interval_ticks),
            _ => {}
        }
    }
    c.leaf_decay_chance_per_second = c.leaf_decay_chance_per_second.clamp(0.0, 1.0);
    c.leaf_check_interval_ticks = c.leaf_check_interval_ticks.max(1);
    c.tree_max_blocks = c.tree_max_blocks.max(4);
    c
}

/// Loads (or creates) `config.properties`. The plugin needs the `fs.read.data` / `fs.write.data` permissions
/// (requested in its metadata; the server owner may have to allow them). The data folder is tried as the
/// server reports it, then as the WASI root, since the guest's view of the preopened folder can differ.
pub fn load(data_folder: &str) {
    let defaults = PluginConfig::default();
    let candidates = [data_folder, ".", "/", "/data"];
    for dir in candidates {
        let file = Path::new(dir).join("config.properties");
        if let Ok(text) = std::fs::read_to_string(&file) {
            CONFIG.with(|x| x.set(parse(&text, defaults)));
            return;
        }
    }
    for dir in candidates {
        let d = Path::new(dir);
        let _ = std::fs::create_dir_all(d);
        if std::fs::write(d.join("config.properties"), describe(&defaults)).is_ok() {
            eprintln!("[physics] created config.properties in '{dir}'");
            break;
        }
    }
    CONFIG.with(|x| x.set(defaults));
}
