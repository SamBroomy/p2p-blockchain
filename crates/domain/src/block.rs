use blake3::{Hash, Hasher};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_valid::Validate;

use crate::transaction::Transaction;

#[derive(Debug, Clone)]
pub struct BlockConstructor {
    index: u64,
    transactions: Vec<Transaction>,
    previous_hash: Hash,
    hash_state: Hasher,
}

impl BlockConstructor {
    pub fn new(index: u64, transactions: &[Transaction], previous_hash: Hash) -> Self {
        let mut hasher = Hasher::new();

        hasher = utils::hash_initial(hasher, index, &previous_hash, transactions);

        Self {
            index,
            transactions: transactions.to_vec(),
            previous_hash,
            hash_state: hasher,
        }
    }

    pub fn mine(self, difficulty: usize, initial_nonce: impl Into<Option<u64>>) -> Block {
        let mut nonce = initial_nonce.into().unwrap_or(0);
        loop {
            let hasher = self.hash_state.clone();
            let hash = utils::hash_with_nonce(hasher, nonce);

            if utils::is_valid_target_hash(&hash, difficulty) {
                let inner =
                    BlockInner::new(self.index, self.transactions, self.previous_hash, nonce);
                return Block::new(inner, hash, difficulty);
            }
            nonce = nonce.wrapping_add(1);
        }
    }
}
/// No assumptions about the validity of the inner block. (The inner block does not have to be valid.)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct BlockInner {
    index: u64,
    transactions: Vec<Transaction>,
    previous_hash: Hash,
    nonce: u64,
}

impl BlockInner {
    pub fn hash(&self) -> Hash {
        utils::hash(
            Hasher::new(),
            self.index,
            &self.previous_hash,
            &self.transactions,
            self.nonce,
        )
    }

    pub fn new(
        index: u64,
        transactions: Vec<Transaction>,
        previous_hash: Hash,
        nonce: u64,
    ) -> Self {
        Self {
            index,
            transactions,
            previous_hash,
            nonce,
        }
    }
}

/// Assumes the block is valid. (In order to create a block, it must be valid.)
#[derive(Debug, Clone, Serialize, Deserialize, Validate, PartialEq, Eq)]
#[validate(custom = utils::validate_block)] // to uphold our invariant that a `Block` struct is always valid (even when deserialized)
pub struct Block {
    #[serde(flatten)]
    inner: BlockInner,
    timestamp: DateTime<Utc>,
    hash: Hash,
    difficulty: usize,
}

impl Block {
    fn new(inner: BlockInner, hash: Hash, difficulty: usize) -> Self {
        assert!(
            utils::is_valid_inner_block(&inner, difficulty),
            "The inner block must be valid in order to create a Block."
        );
        assert!(
            utils::is_valid_block_hash(&hash, &inner),
            "The block hash must match the inner block hash in order to create a Block."
        );
        Self {
            inner,
            timestamp: Utc::now(),
            hash,
            difficulty,
        }
    }

    /// Validates this block was created with the stored difficulty.
    /// Should always return true for properly constructed blocks.
    pub fn is_valid(&self) -> bool {
        utils::is_valid_block(self)
    }

    pub fn hash(&self) -> &Hash {
        &self.hash
    }

    pub fn previous_hash(&self) -> &Hash {
        &self.inner.previous_hash
    }

    pub fn transactions(&self) -> &[Transaction] {
        &self.inner.transactions
    }

    pub fn difficulty(&self) -> usize {
        self.difficulty
    }
}

mod utils {
    use super::{Block, BlockInner, Hash, Hasher, Transaction};

    #[inline]
    pub fn hash_initial(
        mut hasher: Hasher,
        index: u64,
        previous_hash: &Hash,
        transactions: &[Transaction],
    ) -> Hasher {
        hasher
            .update(index.to_le_bytes().as_ref())
            .update(previous_hash.as_bytes());
        for tx in transactions {
            hasher.update(tx.hash().as_bytes());
        }
        hasher
    }

    #[inline]
    pub fn hash_with_nonce(mut hasher: Hasher, nonce: u64) -> Hash {
        hasher.update(nonce.to_le_bytes().as_ref());
        hasher.finalize()
    }

    #[inline]
    pub fn hash(
        hasher: Hasher,
        index: u64,
        previous_hash: &Hash,
        transactions: &[Transaction],
        nonce: u64,
    ) -> Hash {
        let hasher = hash_initial(hasher, index, previous_hash, transactions);
        hash_with_nonce(hasher, nonce)
    }

    #[inline]
    pub fn is_valid_target_hash(hash: &Hash, difficulty: usize) -> bool {
        let bytes = hash.as_bytes();

        // The hash is a hex string, so each byte represents two hex digits.
        let full_zeros = difficulty / 2; // Number of complete zero bytes
        let partial_zero = difficulty % 2; // Remaining hex digit

        for byte in bytes.iter().take(full_zeros) {
            // Check complete zero bytes
            if *byte != 0 {
                // Must be exactly 0x00
                return false;
            }
        }
        if partial_zero > 0 && bytes[full_zeros] >> 4 != 0 {
            // Check remaining hex digit (upper nibble of the next byte)
            return false;
        }
        true
    }

    #[inline]
    pub fn is_valid_inner_block(inner: &BlockInner, difficulty: usize) -> bool {
        is_valid_target_hash(&inner.hash(), difficulty)
    }
    #[inline]
    pub fn is_valid_block_hash(block_hash: &Hash, inner: &BlockInner) -> bool {
        inner.hash() == *block_hash
    }
    #[inline]
    pub fn is_valid_block(block: &Block) -> bool {
        is_valid_inner_block(&block.inner, block.difficulty)
            && is_valid_block_hash(&block.hash, &block.inner)
    }

    pub fn validate_block(block: &Block) -> Result<(), serde_valid::validation::Error> {
        if is_valid_block(block) {
            Ok(())
        } else {
            Err(serde_valid::validation::Error::Custom(
                "Invalid block".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mine_block_with_zero_difficulty() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let constructor = BlockConstructor::new(0, &[], previous_hash);
        let block = constructor.mine(0, None);

        assert!(block.is_valid());
        assert_eq!(block.previous_hash(), &previous_hash);
    }

    #[test]
    fn test_mine_block_with_difficulty() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let constructor = BlockConstructor::new(0, &[], previous_hash);
        let block = constructor.mine(2, None);

        assert!(block.is_valid());
        let hash_bytes = block.hash().as_bytes();
        assert_eq!(hash_bytes[0], 0);
    }

    #[test]
    fn test_block_hash_matches() {
        let previous_hash = Hash::from_bytes([1u8; 32]);
        let constructor = BlockConstructor::new(1, &[], previous_hash);
        let block = constructor.mine(1, None);

        assert_eq!(block.hash(), &block.inner.hash());
    }

    #[test]
    fn test_block_with_transactions() {
        use crate::wallet::Wallet;

        let mut wallet1 = Wallet::new();
        let wallet2 = Wallet::new();
        let tx = wallet1.create_transaction(wallet2.address(), 50);

        let previous_hash = Hash::from_bytes([0u8; 32]);
        let constructor = BlockConstructor::new(0, &[tx], previous_hash);
        let block = constructor.mine(1, None);

        assert!(block.is_valid());
    }

    #[test]
    fn test_block_serialization() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let constructor = BlockConstructor::new(0, &[], previous_hash);
        let block = constructor.mine(1, None);

        let serialized = serde_json::to_string(&block).expect("serialization failed");
        let deserialized: Block =
            serde_json::from_str(&serialized).expect("deserialization failed");

        assert_eq!(block.hash(), deserialized.hash());
        assert_eq!(block.previous_hash(), deserialized.previous_hash());
        assert!(deserialized.is_valid());
    }
}
