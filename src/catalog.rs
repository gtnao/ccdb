//! In-memory catalog of table definitions. day05 hardcodes a `users` table;
//! later days will make this mutable / persistent.

use crate::tuple::{Column as RtColumn, DataType, Schema};

#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
}

#[derive(Debug, Clone)]
pub struct TableDef {
    pub name: String,
    pub columns: Vec<ColumnDef>,
}

impl TableDef {
    pub fn find_column(&self, name: &str) -> Option<(usize, &ColumnDef)> {
        self.columns
            .iter()
            .enumerate()
            .find(|(_, c)| c.name == name)
    }

    pub fn to_schema(&self) -> Schema {
        Schema {
            columns: self
                .columns
                .iter()
                .map(|c| RtColumn {
                    name: c.name.clone(),
                    data_type: c.data_type,
                })
                .collect(),
        }
    }
}

pub struct Catalog {
    tables: Vec<TableDef>,
}

impl Catalog {
    pub fn new() -> Self {
        Self {
            tables: vec![TableDef {
                name: "users".into(),
                columns: vec![
                    ColumnDef {
                        name: "id".into(),
                        data_type: DataType::Int,
                        nullable: false,
                    },
                    ColumnDef {
                        name: "name".into(),
                        data_type: DataType::Varchar,
                        nullable: true,
                    },
                ],
            }],
        }
    }

    pub fn find_table(&self, name: &str) -> Option<(usize, &TableDef)> {
        self.tables
            .iter()
            .enumerate()
            .find(|(_, t)| t.name == name)
    }

    #[allow(dead_code)]
    pub fn table_by_id(&self, id: usize) -> Option<&TableDef> {
        self.tables.get(id)
    }
}
