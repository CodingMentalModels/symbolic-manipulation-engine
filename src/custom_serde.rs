use std::collections::HashMap;

use serde::{
    de::{Deserialize, Deserializer, SeqAccess, Visitor},
    Serialize, Serializer,
};

pub fn deserialize_vector_as_hashmap<'de, D, K, V>(
    deserializer: D,
) -> Result<HashMap<K, V>, D::Error>
where
    D: Deserializer<'de>,
    K: Deserialize<'de> + Eq + std::hash::Hash,
    V: Deserialize<'de>,
{
    struct VecVisitor<K, V> {
        marker: std::marker::PhantomData<fn() -> (K, V)>,
    }

    impl<'de, K, V> Visitor<'de> for VecVisitor<K, V>
    where
        K: Deserialize<'de> + Eq + std::hash::Hash,
        V: Deserialize<'de>,
    {
        type Value = HashMap<K, V>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a sequence of tuples (key, value)")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut map = HashMap::new();
            while let Some((key, value)) = seq.next_element::<(K, V)>()? {
                map.insert(key, value);
            }
            Ok(map)
        }
    }

    let visitor = VecVisitor {
        marker: std::marker::PhantomData,
    };
    deserializer.deserialize_seq(visitor)
}

pub fn serialize_hashmap_as_vector<S, K, V>(
    map: &HashMap<K, V>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    K: Serialize,
    V: Serialize,
{
    let vec: Vec<(&K, &V)> = map.iter().collect();
    vec.serialize(serializer)
}
