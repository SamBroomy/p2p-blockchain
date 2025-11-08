use std::collections::HashMap;

use blake3::Hash;

use crate::Block;
const MAX_ORPHAN_BLOCKS: usize = 100; // Prevent DoS attacks
const MAX_ORPHAN_AGE_SECONDS: u64 = 600; // 10 minutes

/// Tracks blocks that arrived before their parent
#[derive(Debug)]
struct OrphanBlock {
    block: Block,
    arrival_time: std::time::Instant,
}

/// Manages orphaned blocks and requests for missing ancestors
#[derive(Debug)]
pub(super) struct OrphanPool {
    /// Orphan blocks indexed by their parent hash (the hash they need)
    orphans_by_parent: HashMap<Hash, Vec<OrphanBlock>>,
    /// Total count of orphaned blocks
    count: usize,
}

impl OrphanPool {
    pub fn new() -> Self {
        Self {
            orphans_by_parent: HashMap::new(),
            count: 0,
        }
    }

    /// Add an orphan block
    pub fn add_orphan(&mut self, block: Block) -> Result<(), Block> {
        if self.count >= MAX_ORPHAN_BLOCKS {
            return Err(block);
        }

        let parent_hash = *block.previous_hash();
        let orphan = OrphanBlock {
            block,
            arrival_time: std::time::Instant::now(),
        };

        self.orphans_by_parent
            .entry(parent_hash)
            .or_default()
            .push(orphan);

        self.count += 1;
        debug_assert_eq!(
            self.count,
            self.orphans_by_parent.values().map(Vec::len).sum::<usize>(),
            "Orphan count cache is out of sync with actual orphan count"
        );
        Ok(())
    }

    /// Get all orphans that depend on a specific block
    pub fn get_orphans_for_parent(&mut self, parent_hash: &Hash) -> Vec<Block> {
        if let Some(orphans) = self.orphans_by_parent.remove(parent_hash) {
            let orphan_count = orphans.len();
            debug_assert!(
                self.count >= orphan_count,
                "Orphan count {} cannot be less than removed count {orphan_count}",
                self.count,
            );
            self.count = self.count.saturating_sub(orphan_count);

            // ⚡ Debug assertion to verify consistency
            debug_assert_eq!(
                self.count,
                self.orphans_by_parent.values().map(Vec::len).sum::<usize>(),
                "Orphan count cache {} doesn't match actual {}",
                self.count,
                self.orphans_by_parent.values().map(Vec::len).sum::<usize>()
            );

            orphans.into_iter().map(|o| o.block).collect()
        } else {
            Vec::new()
        }
    }

    /// Remove expired orphans
    pub fn prune_expired(&mut self) {
        let now = std::time::Instant::now();

        self.orphans_by_parent.retain(|_, orphans| {
            orphans
                .retain(|o| now.duration_since(o.arrival_time).as_secs() < MAX_ORPHAN_AGE_SECONDS);
            !orphans.is_empty()
        });

        self.count = self.orphans_by_parent.values().map(Vec::len).sum();

        debug_assert_eq!(
            self.count,
            self.orphans_by_parent.values().map(Vec::len).sum::<usize>(),
            "Orphan count cache is out of sync after pruning"
        );
    }

    /// Check if we have orphans waiting for a specific block
    pub fn has_orphans_for(&self, hash: &Hash) -> bool {
        self.orphans_by_parent.contains_key(hash)
    }

    /// Get all missing parent hashes (what we need to request)
    pub fn missing_ancestors(&self) -> Vec<Hash> {
        self.orphans_by_parent.keys().copied().collect()
    }

    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(
            self.count == 0,
            self.orphans_by_parent.is_empty(),
            "Count is {} but orphans_by_parent.is_empty() = {}",
            self.count,
            self.orphans_by_parent.is_empty()
        );
        self.count == 0
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(
            self.count,
            self.orphans_by_parent.values().map(Vec::len).sum::<usize>(),
            "Orphan count cache {} doesn't match actual count",
            self.count
        );
        self.count
    }
}

impl Default for OrphanPool {
    fn default() -> Self {
        Self::new()
    }
}
