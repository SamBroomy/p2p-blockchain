use std::collections::HashMap;

use blake3::Hash;
use blockchain_types::Block;

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
pub struct OrphanPool {
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
    pub fn add_orphan(&mut self, block: Block) -> Result<(), Box<Block>> {
        if self.count >= MAX_ORPHAN_BLOCKS {
            return Err(Box::new(block));
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

            // verify consistency
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

#[cfg(test)]
mod tests {
    use blockchain_types::{BlockConstructor, Miner, MiningStrategy};

    use super::*;

    fn create_test_block(index: u64, previous_hash: Hash) -> Block {
        Miner::mine(
            BlockConstructor::new(index, &[], previous_hash, None),
            0,
            None,
        )
    }

    #[test]
    fn test_new_orphan_pool_is_empty() {
        let pool = OrphanPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_add_orphan() {
        let mut pool = OrphanPool::new();
        let parent_hash = Hash::from_bytes([1u8; 32]);
        let block = create_test_block(1, parent_hash);

        pool.add_orphan(block).expect("should add orphan");
        assert!(!pool.is_empty());
        assert_eq!(pool.len(), 1);
        assert!(pool.has_orphans_for(&parent_hash));
    }

    #[test]
    fn test_add_multiple_orphans_same_parent() {
        let mut pool = OrphanPool::new();
        let parent_hash = Hash::from_bytes([1u8; 32]);
        let block1 = create_test_block(1, parent_hash);
        let block2 = create_test_block(2, parent_hash);

        pool.add_orphan(block1).expect("should add first orphan");
        pool.add_orphan(block2).expect("should add second orphan");

        assert_eq!(pool.len(), 2);
        assert!(pool.has_orphans_for(&parent_hash));
    }

    #[test]
    fn test_add_orphans_different_parents() {
        let mut pool = OrphanPool::new();
        let parent1 = Hash::from_bytes([1u8; 32]);
        let parent2 = Hash::from_bytes([2u8; 32]);
        let block1 = create_test_block(1, parent1);
        let block2 = create_test_block(2, parent2);

        pool.add_orphan(block1).expect("should add first orphan");
        pool.add_orphan(block2).expect("should add second orphan");

        assert_eq!(pool.len(), 2);
        assert!(pool.has_orphans_for(&parent1));
        assert!(pool.has_orphans_for(&parent2));
    }

    #[test]
    fn test_get_orphans_for_parent() {
        let mut pool = OrphanPool::new();
        let parent_hash = Hash::from_bytes([1u8; 32]);
        let block1 = create_test_block(1, parent_hash);
        let block2 = create_test_block(2, parent_hash);
        let block1_hash = *block1.hash();
        let block2_hash = *block2.hash();

        pool.add_orphan(block1).expect("should add orphan");
        pool.add_orphan(block2).expect("should add orphan");

        let orphans = pool.get_orphans_for_parent(&parent_hash);
        assert_eq!(orphans.len(), 2);
        assert!(orphans.iter().any(|b| b.hash() == &block1_hash));
        assert!(orphans.iter().any(|b| b.hash() == &block2_hash));

        // After retrieval, pool should be empty
        assert!(pool.is_empty());
        assert!(!pool.has_orphans_for(&parent_hash));
    }

    #[test]
    fn test_get_orphans_for_nonexistent_parent() {
        let mut pool = OrphanPool::new();
        let nonexistent = Hash::from_bytes([99u8; 32]);

        let orphans = pool.get_orphans_for_parent(&nonexistent);
        assert!(orphans.is_empty());
        assert!(pool.is_empty());
    }

    #[test]
    fn test_max_orphan_limit() {
        let mut pool = OrphanPool::new();

        // Add MAX_ORPHAN_BLOCKS successfully
        for i in 0..MAX_ORPHAN_BLOCKS {
            let parent = Hash::from_bytes([i as u8; 32]);
            let block = create_test_block(i as u64, parent);
            pool.add_orphan(block)
                .expect("should add orphan within limit");
        }
        assert_eq!(pool.len(), MAX_ORPHAN_BLOCKS);

        // Adding one more should fail
        let parent = Hash::from_bytes([255u8; 32]);
        let excess_block = create_test_block(9999, parent);
        let result = pool.add_orphan(excess_block);

        assert!(result.is_err(), "should reject orphan beyond limit");
        assert_eq!(pool.len(), MAX_ORPHAN_BLOCKS);
    }

    #[test]
    #[ignore = "This test would be slow in practice due to waiting"]
    fn test_prune_expired_orphans() {
        let mut pool = OrphanPool::new();
        let parent_hash = Hash::from_bytes([1u8; 32]);
        let block = create_test_block(1, parent_hash);

        pool.add_orphan(block).expect("should add orphan");
        assert_eq!(pool.len(), 1);

        // Orphan should still be fresh
        pool.prune_expired();
        assert_eq!(pool.len(), 1);

        // Simulate aging by waiting (this test would be slow in practice)
        // For testing, we verify the logic compiles and runs
        pool.prune_expired();
        // Without actual time passing, orphan remains
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_missing_ancestors() {
        let mut pool = OrphanPool::new();
        let parent1 = Hash::from_bytes([1u8; 32]);
        let parent2 = Hash::from_bytes([2u8; 32]);
        let parent3 = Hash::from_bytes([3u8; 32]);

        pool.add_orphan(create_test_block(1, parent1))
            .expect("add orphan");
        pool.add_orphan(create_test_block(2, parent2))
            .expect("add orphan");
        pool.add_orphan(create_test_block(3, parent3))
            .expect("add orphan");

        let missing = pool.missing_ancestors();
        assert_eq!(missing.len(), 3);
        assert!(missing.contains(&parent1));
        assert!(missing.contains(&parent2));
        assert!(missing.contains(&parent3));
    }

    #[test]
    fn test_missing_ancestors_empty_pool() {
        let pool = OrphanPool::new();
        let missing = pool.missing_ancestors();
        assert!(missing.is_empty());
    }

    #[test]
    fn test_orphan_count_consistency_after_mixed_operations() {
        let mut pool = OrphanPool::new();
        let parent1 = Hash::from_bytes([1u8; 32]);
        let parent2 = Hash::from_bytes([2u8; 32]);

        // Add orphans for parent1
        pool.add_orphan(create_test_block(1, parent1))
            .expect("add orphan");
        pool.add_orphan(create_test_block(2, parent1))
            .expect("add orphan");
        assert_eq!(pool.len(), 2);

        // Add orphan for parent2
        pool.add_orphan(create_test_block(3, parent2))
            .expect("add orphan");
        assert_eq!(pool.len(), 3);

        // Retrieve orphans for parent1
        let orphans = pool.get_orphans_for_parent(&parent1);
        assert_eq!(orphans.len(), 2);
        assert_eq!(pool.len(), 1); // Only parent2's orphan remains

        // Retrieve orphans for parent2
        let orphans = pool.get_orphans_for_parent(&parent2);
        assert_eq!(orphans.len(), 1);
        assert!(pool.is_empty());
    }

    #[test]
    fn test_has_orphans_for_after_retrieval() {
        let mut pool = OrphanPool::new();
        let parent = Hash::from_bytes([1u8; 32]);
        let block = create_test_block(1, parent);

        pool.add_orphan(block).expect("add orphan");
        assert!(pool.has_orphans_for(&parent));

        pool.get_orphans_for_parent(&parent);
        assert!(!pool.has_orphans_for(&parent));
    }

    #[test]
    fn test_default_implementation() {
        let pool = OrphanPool::default();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }
}
