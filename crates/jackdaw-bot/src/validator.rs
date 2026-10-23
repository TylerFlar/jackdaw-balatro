//! The schema validator: fail-fast type checks of an endpoint's arguments.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/core/validator.lua`, branch for branch, with Lua semantics: a JSON `null` is an absent
//! field, `{}` and `[]` are both empty tables, "integer" means `math.floor(v) == v`.
//!
//! `Validator.validate` walks `pairs(schema)`, so with several bad fields the first one *in the
//! table's iteration order* is reported. Each endpoint lists its fields in that order: the order
//! Balatro's LuaJIT 2.0.5 (`lua51.dll`) gives `pairs` over the schema table its constructor builds
//! (`scripts/luajit205_probe.py schema-order` prints it from the game's own runtime).

use crate::errors::BotError;
use crate::lua::{self, Table, Value};

/// `Endpoint.Schema.type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldType {
    String,
    Integer,
    Boolean,
    Array,
    Table,
}

impl FieldType {
    pub fn name(self) -> &'static str {
        match self {
            FieldType::String => "string",
            FieldType::Integer => "integer",
            FieldType::Boolean => "boolean",
            FieldType::Array => "array",
            FieldType::Table => "table",
        }
    }
}

/// One schema field (`schema[name] = {type, required, items}`).
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub name: &'static str,
    pub ty: FieldType,
    pub required: bool,
    /// `items`: the element type of an array (BalatroBot only ever uses `"integer"`).
    pub items: Option<FieldType>,
}

impl Field {
    pub const fn new(name: &'static str, ty: FieldType, required: bool) -> Field {
        Field {
            name,
            ty,
            required,
            items: None,
        }
    }

    pub const fn array_of(name: &'static str, items: FieldType, required: bool) -> Field {
        Field {
            name,
            ty: FieldType::Array,
            required,
            items: Some(items),
        }
    }
}

/// `validate_field(field_name, value, field_schema)` (validator.lua:37-68).
fn validate_field(field: &Field, value: &Value) -> Result<(), BotError> {
    let name = field.name;
    match field.ty {
        FieldType::Integer => {
            if !lua::is_integer(Some(value)) {
                return Err(BotError::bad_request(format!(
                    "Field '{name}' must be an integer"
                )));
            }
        }
        FieldType::Array => {
            if !value.as_table().is_some_and(Table::is_array) {
                return Err(BotError::bad_request(format!(
                    "Field '{name}' must be an array"
                )));
            }
        }
        FieldType::Table => {
            let ok = value
                .as_table()
                .is_some_and(|t| t.is_empty() || !t.is_array());
            if !ok {
                return Err(BotError::bad_request(format!(
                    "Field '{name}' must be a table"
                )));
            }
        }
        FieldType::String | FieldType::Boolean => {
            if value.type_name() != field.ty.name() {
                return Err(BotError::bad_request(format!(
                    "Field '{name}' must be of type {}",
                    field.ty.name()
                )));
            }
        }
    }
    if let (FieldType::Array, Some(items), Some(t)) = (field.ty, field.items, value.as_table()) {
        for (i, item) in t.ipairs().enumerate() {
            let ok = match items {
                FieldType::Integer => lua::is_integer(Some(item)),
                other => item.type_name() == other.name(),
            };
            if !ok {
                return Err(BotError::bad_request(format!(
                    "Field '{name}' array item at index {i} must be of type {}",
                    items.name()
                )));
            }
        }
    }
    Ok(())
}

/// `Validator.validate(args, schema)` (validator.lua:75-92). `None` is a non-table argument.
pub fn validate(args: Option<&Table>, schema: &[Field]) -> Result<(), BotError> {
    let Some(args) = args else {
        return Err(BotError::bad_request("Arguments must be a table"));
    };
    for field in schema {
        let value = args.get(field.name);
        if field.required && value.is_none() {
            return Err(BotError::bad_request(format!(
                "Missing required field '{}'",
                field.name
            )));
        }
        if let Some(value) = value {
            validate_field(field, value)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::decode;

    fn table(text: &str) -> Table {
        match decode(text.as_bytes()).unwrap() {
            Some(Value::Table(t)) => t,
            other => panic!("not a table: {other:?}"),
        }
    }

    const SCHEMA: &[Field] = &[
        Field::new("t", FieldType::Table, false),
        Field::new("i", FieldType::Integer, false),
        Field::array_of("a", FieldType::Integer, false),
        Field::new("s", FieldType::String, true),
        Field::new("b", FieldType::Boolean, false),
    ];

    fn check(args: &str) -> Result<(), String> {
        validate(Some(&table(args)), SCHEMA).map_err(|e| e.message)
    }

    #[test]
    fn lua_semantics() {
        assert_eq!(check(r#"{"s": "x"}"#), Ok(()));
        assert_eq!(check(r#"{"s": "x", "i": 2.0, "t": [], "a": []}"#), Ok(()));
        assert_eq!(check(r#"{"s": "x", "t": {"k": 1}}"#), Ok(()));
        assert_eq!(check(r#"{"s": "x", "i": null}"#), Ok(()), "null is absent");
        assert_eq!(
            check(r#"{"s": null}"#),
            Err("Missing required field 's'".into())
        );
        assert_eq!(
            check(r#"{"s": "x", "i": 1.5}"#),
            Err("Field 'i' must be an integer".into())
        );
        assert_eq!(
            check(r#"{"s": "x", "t": [1]}"#),
            Err("Field 't' must be a table".into())
        );
        assert_eq!(
            check(r#"{"s": "x", "a": [1, null, 3]}"#),
            Err("Field 'a' must be an array".into())
        );
        assert_eq!(
            check(r#"{"s": "x", "a": {"0": 1}}"#),
            Err("Field 'a' must be an array".into())
        );
        assert_eq!(
            check(r#"{"s": "x", "a": [1, 2.5]}"#),
            Err("Field 'a' array item at index 1 must be of type integer".into())
        );
        assert_eq!(
            check(r#"{"s": 1}"#),
            Err("Field 's' must be of type string".into())
        );
        assert_eq!(
            check(r#"{"s": "x", "b": 0}"#),
            Err("Field 'b' must be of type boolean".into())
        );
        // Fail fast in schema order: `t` comes before the missing `s`.
        assert_eq!(
            check(r#"{"t": 5}"#),
            Err("Field 't' must be a table".into())
        );
        assert_eq!(
            validate(None, SCHEMA).map_err(|e| e.message),
            Err("Arguments must be a table".into())
        );
    }
}
