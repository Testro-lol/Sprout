//! Custom liquids: material definitions the user made, saved in a versioned file.
//!
//! A definition is plain data with a stable id (`custom-...`); built-ins have the ids
//! `builtin.<key>` and are read-only. The file is JSON under the XDG config directory
//! (`$XDG_CONFIG_HOME/sprout/liquids.json`, or `~/.config/sprout/liquids.json`;
//! `$SPROUT_LIBRARY` overrides the path). Every numeric property is written and read
//! through the parameter table (`MAT_PARAMS`): the same names, units and ranges as
//! `--set` and the panel. On load a value outside its range is clamped and a missing or
//! unreadable one is taken from the liquid's base preset, each with a warning; a file of
//! another format version is read the same way, with a warning.
//!
//! Dependency: `serde_json` (the parser; the format itself is defined here, by the table).

use std::path::PathBuf;

use serde_json::{Map, Value, json};

use crate::sim::{BUILTINS, MAT_PARAMS, MAX_MATERIALS, MatProps, PhaseGroup, SimParams};

/// Version of the file format written now.
pub const FORMAT_VERSION: u64 = 1;

/// One custom liquid.
#[derive(Clone, Debug, PartialEq)]
pub struct CustomLiquid {
    /// Stable identity, kept across sessions and renames.
    pub id: String,
    pub name: String,
    /// Built-in it started from (`BUILTINS` key): the defaults for anything the file lacks.
    pub base: String,
    pub props: MatProps,
    /// Deleted this session: left out of the file and the picker; its row keeps its
    /// definition until the tank is emptied (rows are not reused within a session).
    pub deleted: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Library {
    pub custom: Vec<CustomLiquid>,
    pub path: PathBuf,
    /// What loading had to fix or could not read.
    pub warnings: Vec<String>,
}

/// Row of the live table that custom liquid `k` occupies.
pub fn custom_row(k: usize) -> usize {
    BUILTINS.len() + k
}

/// Most custom liquids a session can hold (rows of the live table).
pub const MAX_CUSTOM: usize = MAX_MATERIALS - BUILTINS.len();

fn group_name(g: u32) -> &'static str {
    match PhaseGroup::from_index(g) {
        PhaseGroup::Aqueous => "aqueous",
        PhaseGroup::Oily => "oil-like",
        PhaseGroup::Metal => "metal",
        PhaseGroup::Silicate => "silicate",
    }
}

fn group_of(name: &str) -> Option<u32> {
    ["aqueous", "oil-like", "metal", "silicate"].iter().position(|g| *g == name).map(|g| g as u32)
}

impl Library {
    /// Where the library lives.
    pub fn default_path() -> PathBuf {
        if let Some(p) = std::env::var_os("SPROUT_LIBRARY") {
            return PathBuf::from(p);
        }
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("sprout").join("liquids.json")
    }

    /// Reads the library at `path`; a missing file is an empty library.
    pub fn load(path: PathBuf) -> Library {
        let mut lib = Library { path, ..Default::default() };
        match std::fs::read_to_string(&lib.path) {
            Ok(text) => {
                let (custom, warnings) = parse(&text);
                lib.custom = custom;
                lib.warnings = warnings;
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => lib.warnings.push(format!("cannot read {}: {e}", lib.path.display())),
        }
        lib
    }

    /// Writes the library to its file (creating the directory).
    pub fn save(&self) -> Result<(), String> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let text = serde_json::to_string_pretty(&self.to_json()).map_err(|e| e.to_string())?;
        // Write then rename, so that a crash cannot leave half a file.
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, text).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| format!("cannot write {}: {e}", self.path.display()))
    }

    /// Removes one liquid from the file on disk, leaving the rest of the file as it is.
    pub fn remove_from_file(&self, id: &str) -> Result<(), String> {
        let mut on_disk = Library::load(self.path.clone());
        let before = on_disk.custom.len();
        on_disk.custom.retain(|c| c.id != id);
        if on_disk.custom.len() == before {
            return Ok(());
        }
        on_disk.save()
    }

    pub fn to_json(&self) -> Value {
        let ids: Vec<String> = self.custom.iter().map(|c| c.id.clone()).collect();
        json!({
            "format": "sprout-liquids",
            "version": FORMAT_VERSION,
            "liquids": self.custom.iter().filter(|c| !c.deleted).map(|c| liquid_to_json(c, &ids)).collect::<Vec<_>>(),
        })
    }

    /// Puts every custom liquid into its row of the live table (`custom_row`), after the
    /// built-ins, and sets how many rows are in use.
    pub fn install(&self, p: &mut SimParams) {
        let n = self.custom.len().min(MAX_CUSTOM);
        for (k, c) in self.custom.iter().take(n).enumerate() {
            p.mat[custom_row(k)] = c.props;
        }
        p.mat_count = p.mat_count.max(BUILTINS.len() + n);
    }

    /// Table row of a liquid by id (`builtin.<key>` or a custom id).
    #[cfg(test)]
    pub fn row_of(&self, id: &str) -> Option<usize> {
        if let Some(key) = id.strip_prefix("builtin.") {
            return BUILTINS.iter().position(|b| b.key == key);
        }
        self.custom.iter().position(|c| c.id == id).map(custom_row)
    }

    /// Display name of a table row.
    pub fn name(&self, row: usize) -> String {
        if row < BUILTINS.len() {
            return BUILTINS[row].name.to_string();
        }
        self.custom.get(row - BUILTINS.len()).map_or_else(|| format!("Liquid {row}"), |c| c.name.clone())
    }

    /// A new id no liquid has: the time plus a counter.
    pub fn new_id(&self) -> String {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_micros());
        let mut k = 0;
        loop {
            let id = format!("custom-{t:x}-{k}");
            if self.custom.iter().all(|c| c.id != id) {
                return id;
            }
            k += 1;
        }
    }
}

/// Keys of built-ins that were renamed on 2026-11-16, for files saved before.
pub fn renamed(key: &str) -> &str {
    match key {
        "syrup" => "honey",
        // Wet sand was removed on 2026-11-16 (sand will be a solid of its own): its files load as mud.
        "sand" => "mud",
        "sludge" => "clay",
        // The marbled liquid went with the made-up looks on 2026-11-17; its row is the snow globe.
        "marbled" => "snowy",
        k => k,
    }
}

/// Reference to a table row in the file: a built-in's id or a custom liquid's.
fn row_ref(row: i32, ids: &[String]) -> Value {
    if row < 0 {
        return Value::Null;
    }
    let row = row as usize;
    if row < BUILTINS.len() {
        Value::String(format!("builtin.{}", BUILTINS[row].key))
    } else {
        ids.get(row - BUILTINS.len()).map_or(Value::Null, |id| Value::String(id.clone()))
    }
}

fn liquid_to_json(c: &CustomLiquid, ids: &[String]) -> Value {
    let mut props = Map::new();
    let mut scratch = SimParams::default();
    scratch.mat[0] = c.props;
    for p in MAT_PARAMS {
        let v = (p.get)(&scratch, 0);
        props.insert(p.key.to_string(), json!(v));
    }
    props.insert("group".into(), json!(group_name(c.props.group)));
    props.insert("diluent".into(), row_ref(c.props.diluent, ids));
    props.insert("residue".into(), row_ref(c.props.residue, ids));
    props.insert("color".into(), json!(c.props.color));
    props.insert("color_solid".into(), json!(c.props.color_solid));
    props.insert("glow_color".into(), json!(c.props.glow_color));
    props.insert("color_hot".into(), json!(c.props.color_hot));
    props.insert("cloud_color".into(), json!(c.props.cloud_color));
    props.insert("dust_color".into(), json!(c.props.dust_color));
    props.insert("react_with".into(), row_ref(c.props.react_with, ids));
    props.insert("react_into".into(), row_ref(c.props.react_into, ids));
    json!({ "id": c.id, "name": c.name, "base": c.base, "props": Value::Object(props) })
}

fn colour(v: Option<&Value>) -> Option<[f32; 3]> {
    let a = v?.as_array()?;
    if a.len() != 3 {
        return None;
    }
    let mut c = [0.0; 3];
    for (o, x) in c.iter_mut().zip(a) {
        *o = x.as_f64()? as f32;
    }
    Some(c)
}

/// Parses a library file: the liquids it could read, and warnings.
pub fn parse(text: &str) -> (Vec<CustomLiquid>, Vec<String>) {
    let mut warnings = Vec::new();
    let root: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => return (Vec::new(), vec![format!("the liquids file is not valid JSON ({e}); nothing loaded")]),
    };
    let version = root.get("version").and_then(Value::as_u64);
    if version != Some(FORMAT_VERSION) {
        warnings.push(format!(
            "the liquids file is format version {}, this build writes {FORMAT_VERSION}: read what could be, with defaults for the rest",
            version.map_or("unknown".to_string(), |v| v.to_string())
        ));
    }
    let Some(list) = root.get("liquids").and_then(Value::as_array) else {
        warnings.push("the liquids file has no list of liquids".into());
        return (Vec::new(), warnings);
    };
    // First pass: ids and bases, so that references between custom liquids resolve.
    let mut out: Vec<CustomLiquid> = Vec::new();
    let mut refs: Vec<(Option<String>, Option<String>)> = Vec::new();
    let mut reacts: Vec<(bool, Option<String>, Option<String>)> = Vec::new();
    for (k, item) in list.iter().enumerate().take(MAX_CUSTOM) {
        let id = item.get("id").and_then(Value::as_str).map(str::to_string);
        let name = item.get("name").and_then(Value::as_str).unwrap_or("Unnamed").to_string();
        let id = match id {
            Some(id) if !id.starts_with("builtin.") && out.iter().all(|c| c.id != id) => id,
            _ => {
                warnings.push(format!("liquid {} (\"{name}\") has no usable id: given a new one", k + 1));
                format!("custom-loaded-{k}")
            }
        };
        let base = item.get("base").and_then(Value::as_str).unwrap_or("water").to_string();
        let base_row = BUILTINS.iter().position(|b| b.key == renamed(&base)).unwrap_or_else(|| {
            warnings.push(format!("\"{name}\": unknown base \"{base}\", using water"));
            0
        });
        let mut props = (BUILTINS[base_row].props)();
        let obj = item.get("props").and_then(Value::as_object);
        let mut scratch = SimParams::default();
        scratch.mat[0] = props;
        for p in MAT_PARAMS {
            match obj.and_then(|o| o.get(p.key)).and_then(Value::as_f64) {
                Some(v) => {
                    let v = v as f32;
                    let c = p.clamp(v);
                    if c != v {
                        warnings.push(format!("\"{name}\": {} = {v} is outside {} .. {}, clamped to {c}", p.key, p.min, p.max));
                    }
                    (p.put)(&mut scratch, 0, c);
                }
                None => warnings.push(format!("\"{name}\": {} missing, taken from {}", p.key, BUILTINS[base_row].name)),
            }
        }
        props = scratch.mat[0];
        if let Some(g) = obj.and_then(|o| o.get("group")).and_then(Value::as_str) {
            match group_of(g) {
                Some(g) => props.group = g,
                None => warnings.push(format!("\"{name}\": unknown phase group \"{g}\"")),
            }
        }
        for (field, value) in [("color", &mut props.color), ("color_solid", &mut props.color_solid)] {
            match colour(obj.and_then(|o| o.get(field))) {
                Some(c) => *value = c.map(|x| x.clamp(0.0, 1.0)),
                None => warnings.push(format!("\"{name}\": {field} missing or not three numbers")),
            }
        }
        // (The colours of 2026-11-16; a file without them keeps its base's.)
        for (field, value) in [("glow_color", &mut props.glow_color), ("color_hot", &mut props.color_hot), ("cloud_color", &mut props.cloud_color), ("dust_color", &mut props.dust_color)] {
            if let Some(c) = colour(obj.and_then(|o| o.get(field))) {
                *value = c.map(|x| x.clamp(0.0, 1.0));
            }
        }
        // (`color_mid`, the colour of a half-diluted concentrate, was dropped on 2026-11-14:
        // mixtures now take their colour from absorption, and files that have it ignore it.)
        let reference = |key: &str| obj.and_then(|o| o.get(key)).and_then(Value::as_str).map(str::to_string);
        refs.push((reference("diluent"), reference("residue")));
        reacts.push((obj.is_some_and(|o| o.contains_key("react_with")), reference("react_with"), reference("react_into")));
        out.push(CustomLiquid { id, name, base, props, deleted: false });
    }
    // Second pass: diluent and residue references, to built-ins or to liquids in the file.
    let ids: Vec<String> = out.iter().map(|c| c.id.clone()).collect();
    let resolve = |r: &Option<String>| -> Option<i32> {
        let r = r.as_ref()?;
        if let Some(key) = r.strip_prefix("builtin.") {
            let key = renamed(key);
            return BUILTINS.iter().position(|b| b.key == key).map(|i| i as i32);
        }
        ids.iter().position(|id| id == r).map(|k| custom_row(k) as i32)
    };
    for (k, (dil, res)) in refs.iter().enumerate() {
        let me = custom_row(k) as i32;
        let d = resolve(dil);
        if dil.is_some() && d.is_none() {
            warnings.push(format!("\"{}\": diluent {} not found, it has none", out[k].name, dil.as_ref().unwrap()));
        }
        // A diluent must be a plain liquid: not this one, and not itself a concentrate.
        let plain = |r: i32| (r as usize) < BUILTINS.len() && (BUILTINS[r as usize].props)().diluent < 0;
        out[k].props.diluent = match d {
            Some(r) if r != me && (plain(r) || (r as usize) >= BUILTINS.len()) => r,
            _ => -1,
        };
        out[k].props.residue = resolve(res).unwrap_or(-1);
        // A reaction: as the file has it; a file from before reactions keeps its base's.
        let (has, with, into) = &reacts[k];
        if *has {
            out[k].props.react_with = resolve(with).filter(|&r| r != me).unwrap_or(-1);
            out[k].props.react_into = resolve(into).unwrap_or(-1);
        }
    }
    (out, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Library {
        let mut honey = (BUILTINS[1].props)();
        honey.viscosity = 0.2;
        honey.color = [0.9, 0.6, 0.1];
        let mut sea = (BUILTINS[14].props)();
        sea.diluent = 0;
        Library {
            custom: vec![
                CustomLiquid { id: "custom-a".into(), name: "Honey".into(), base: "honey".into(), props: honey, deleted: false },
                CustomLiquid { id: "custom-b".into(), name: "Sea water".into(), base: "brine".into(), props: sea, deleted: false },
            ],
            path: PathBuf::new(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn a_library_survives_writing_and_reading() {
        let lib = sample();
        let text = serde_json::to_string(&lib.to_json()).unwrap();
        let (back, warnings) = parse(&text);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(back, lib.custom);
    }

    #[test]
    fn bad_values_are_clamped_and_missing_ones_defaulted_with_warnings() {
        let text = r#"{"version": 0, "liquids": [{"id": "x", "name": "Odd", "base": "oil",
            "props": {"density": 1e9, "viscosity": -3, "group": "plasma", "color": [2, 0.5, 0.5]}}]}"#;
        let (back, warnings) = parse(text);
        assert_eq!(back.len(), 1);
        let p = back[0].props;
        assert_eq!(p.density, 14000.0);
        assert!(p.viscosity > 0.0);
        assert_eq!(p.group, PhaseGroup::Oily as u32, "falls back to the base's group");
        assert_eq!(p.color, [1.0, 0.5, 0.5]);
        assert!(warnings.iter().any(|w| w.contains("version")));
        assert!(warnings.iter().any(|w| w.contains("density") && w.contains("clamped")));
        assert!(warnings.iter().any(|w| w.contains("missing")));
    }

    #[test]
    fn garbage_loads_nothing_with_a_warning() {
        let (back, warnings) = parse("not json");
        assert!(back.is_empty() && !warnings.is_empty());
    }

    #[test]
    fn built_in_ids_cannot_be_taken_by_custom_liquids() {
        let text = r#"{"version": 1, "liquids": [{"id": "builtin.water", "name": "Fake water", "base": "water", "props": {}}]}"#;
        let (back, warnings) = parse(text);
        assert_ne!(back[0].id, "builtin.water");
        assert!(warnings.iter().any(|w| w.contains("id")));
        let lib = Library { custom: back, ..Default::default() };
        assert_eq!(lib.row_of("builtin.water"), Some(0), "the built-in keeps its row");
    }
}
