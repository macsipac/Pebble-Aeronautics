//! Real-world-ish material properties for vanilla blocks.
//!
//! Everything here is a best-effort heuristic keyed on the block's registry
//! name, so it keeps working when Mojang adds new blocks (unknown blocks simply
//! fall back to a generic "rock-ish" material).

/// Physical properties of a block used by the physics engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    /// Density in kg/m³ (a block is exactly 1 m³).
    pub density: f32,
    /// Bounciness, 0 = dead stop, 1 = perfectly elastic.
    pub restitution: f32,
}

/// Density of water in kg/m³.
pub const WATER_DENSITY: f32 = 1000.0;
/// Density of lava in kg/m³ (molten basalt is ~2700-3100).
pub const LAVA_DENSITY: f32 = 3100.0;

/// Converts vanilla block `slipperiness` (0.6 normal, 0.98 ice, 0.989 blue ice,
/// 0.8 slime) to a Coulomb friction coefficient.
///
/// `0.6 -> 0.70` (stone on stone), `0.98 -> 0.035` (ice).
#[must_use]
pub fn friction_from_slipperiness(slipperiness: f32) -> f32 {
    (1.75 * (1.0 - slipperiness)).clamp(0.02, 1.2)
}

fn has(name: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| name.contains(n))
}

/// Returns the material for a block registry name (with or without the
/// `minecraft:` prefix).
#[must_use]
pub fn material_for(block_name: &str) -> Material {
    let name = block_name.strip_prefix("minecraft:").unwrap_or(block_name);

    // --- special cases first -------------------------------------------------
    if name == "slime_block" {
        return Material { density: 1100.0, restitution: 0.85 };
    }
    if name == "honey_block" {
        return Material { density: 1400.0, restitution: 0.0 };
    }

    // --- metals / gems (very dense) ------------------------------------------
    let density = if has(name, &["gold"]) {
        19_300.0
    } else if has(name, &["netherite"]) {
        8_000.0
    } else if has(name, &["copper"]) {
        8_960.0
    } else if has(name, &["iron", "anvil", "chain", "cauldron", "hopper"]) {
        7_870.0
    } else if has(name, &["diamond"]) {
        3_510.0
    } else if has(name, &["emerald"]) {
        2_760.0
    } else if has(name, &["lapis", "redstone_block", "amethyst"]) {
        2_700.0
    } else if has(name, &["bedrock", "obsidian"]) {
        3_000.0
    // --- very light --------------------------------------------------------
    } else if has(name, &["hay_block", "sponge", "wool", "carpet", "leaves", "shroomlight", "sculk"]) {
        150.0
    } else if has(name, &["powder_snow", "snow"]) && name != "snow_block" {
        100.0
    } else if name == "snow_block" {
        300.0
    } else if has(name, &["moss", "dried_kelp", "target", "beehive", "bee_nest", "honeycomb"]) {
        350.0
    // --- ice -----------------------------------------------------------------
    } else if has(name, &["ice"]) {
        917.0
    // --- glass ---------------------------------------------------------------
    } else if has(name, &["glass"]) {
        2_500.0
    // --- wood family ---------------------------------------------------------
    } else if has(
        name,
        &[
            "planks", "log", "wood", "stem", "hyphae", "bamboo", "crafting_table", "bookshelf",
            "chest", "barrel", "note_block", "jukebox", "loom", "lectern", "composter", "fletching",
            "smithing", "cartography", "beehive", "pumpkin", "melon", "cactus", "door", "fence",
            "stairs_oak", "slab_oak",
        ],
    ) && !has(name, &["stone", "brick"])
    {
        700.0
    // --- loose ground --------------------------------------------------------
    } else if has(name, &["dirt", "grass_block", "podzol", "mycelium", "mud", "farmland", "path", "clay", "gravel", "sand", "soul_soil", "nylium", "rooted"])
        && !has(name, &["sandstone", "brick"])
    {
        1_600.0
    // --- stone and the rest ----------------------------------------------------
    } else if has(name, &["netherrack"]) {
        1_500.0
    } else if has(name, &["tnt"]) {
        1_000.0
    } else if has(name, &["glowstone", "sea_lantern", "magma"]) {
        1_300.0
    } else if has(name, &["bone"]) {
        1_900.0
    } else if has(name, &["concrete", "terracotta", "brick", "sandstone", "end_stone", "prismarine", "purpur", "quartz"]) {
        2_200.0
    } else {
        // stone, cobblestone, granite, diorite, andesite, deepslate, tuff, ...
        2_600.0
    };

    let restitution = if has(name, &["dirt", "grass_block", "mud", "sand", "gravel", "clay", "snow", "podzol", "farmland"]) {
        0.05
    } else if has(name, &["wool", "hay", "leaves", "carpet", "sponge", "moss"]) {
        0.12
    } else if has(name, &["planks", "log", "wood"]) {
        0.3
    } else if has(name, &["glass", "ice"]) {
        0.25
    } else {
        0.2
    };

    Material { density, restitution }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn friction_mapping() {
        assert!((friction_from_slipperiness(0.6) - 0.7).abs() < 1e-3);
        assert!(friction_from_slipperiness(0.98) < 0.05);
        assert!(friction_from_slipperiness(0.989) < friction_from_slipperiness(0.98));
    }

    #[test]
    fn densities_are_sane() {
        // Things that must float in water (1000):
        for n in ["minecraft:oak_planks", "minecraft:hay_block", "minecraft:ice", "minecraft:oak_log"] {
            assert!(material_for(n).density < WATER_DENSITY, "{n} should float");
        }
        // Things that must sink:
        for n in ["minecraft:stone", "minecraft:dirt", "minecraft:iron_block", "minecraft:gold_block", "minecraft:glass", "minecraft:sandstone"] {
            assert!(material_for(n).density > WATER_DENSITY, "{n} should sink");
        }
        assert!(material_for("minecraft:slime_block").restitution > 0.8);
        assert!(material_for("minecraft:gold_block").density > material_for("minecraft:iron_block").density);
    }
}
