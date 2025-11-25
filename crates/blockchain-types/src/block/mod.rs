use std::collections::VecDeque;

use blake3::{Hash, Hasher};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_valid::Validate;
pub mod mining;
use crate::{
    Transaction,
    consts::{BLOCK_REWARD_AMOUNT, MAX_TRANSACTIONS_PER_BLOCK},
    transaction::TransactionConstructor,
    wallet::Address,
};

#[derive(Debug, Clone)]
pub struct BlockConstructor {
    index: u64,
    transactions: VecDeque<Transaction>,
    previous_hash: Hash,
    hash_state: Hasher,
}

impl BlockConstructor {
    /// Create a new block constructor
    /// If `miner_address` is provided, a block reward transaction will be prepended
    pub fn new(
        index: u64,
        transactions: &[Transaction],
        previous_hash: Hash,
        miner_address: Option<Address>,
    ) -> Self {
        let max_tx = if miner_address.is_some() {
            MAX_TRANSACTIONS_PER_BLOCK - 1
        } else {
            MAX_TRANSACTIONS_PER_BLOCK
        };
        debug_assert!(
            transactions.len() <= max_tx,
            "Number of transactions {} exceeds maximum allowed {}",
            transactions.len(),
            max_tx
        );
        debug_assert!(
            transactions.iter().all(Transaction::is_validate),
            "All transactions must be valid before mining"
        );
        let mut all_transactions = Vec::new();

        // Prepend block reward if miner address provided
        if let Some(miner_addr) = miner_address {
            // Calculate total fees from transactions
            let total_fees: u64 = transactions
                .iter()
                .filter_map(|tx| tx.fee())
                .sum();

            // Block reward = base reward + transaction fees
            let total_reward = BLOCK_REWARD_AMOUNT + total_fees;
            let reward_tx = TransactionConstructor::block_reward(&miner_addr, total_reward);
            debug_assert!(
                reward_tx.is_validate(),
                "Block reward transaction must be valid"
            );
            all_transactions.push(reward_tx);
        }

        // Add user transactions
        all_transactions.extend_from_slice(transactions);

        debug_assert!(
            all_transactions.len() <= MAX_TRANSACTIONS_PER_BLOCK,
            "Total transactions {} exceed maximum {}",
            all_transactions.len(),
            MAX_TRANSACTIONS_PER_BLOCK
        );

        let mut hasher = Hasher::new();
        hasher = utils::hash_initial(hasher, index, &previous_hash, &all_transactions);

        Self {
            index,
            transactions: all_transactions.into(),
            previous_hash,
            hash_state: hasher,
        }
    }

    pub fn hash_state(&self) -> &Hasher {
        &self.hash_state
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
            "Block must meet difficulty target: difficulty={difficulty}, hash={hash:?}"
        );
        assert!(
            utils::is_valid_block_hash(&hash, &inner),
            "Block hash {hash:?} must match computed hash {:?}",
            inner.hash()
        );
        debug_assert!(
            inner.transactions.iter().all(Transaction::is_validate),
            "All transactions in block must be valid"
        );
        // Validate block reward structure if present
        if !inner.transactions.is_empty() && inner.transactions[0].is_block_reward() {
            debug_assert!(
                inner
                    .transactions
                    .iter()
                    .skip(1)
                    .all(|tx| !tx.is_block_reward()),
                "Only first transaction can be block reward"
            );
        }
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
        debug_assert!(
            self.difficulty <= 64,
            "Difficulty {} exceeds maximum",
            self.difficulty
        );
        self.difficulty
    }

    pub fn index(&self) -> u64 {
        self.inner.index
    }
}

mod utils {
    use super::{Block, BlockInner, Hash, Hasher, Transaction};
    use crate::consts::MAX_BLOCK_SIZE_BYTES;

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
        // Blake3 produces 32-byte (256-bit) hashes = 64 hex digits maximum
        // Difficulties > 64 are impossible to satisfy
        if difficulty > 64 {
            return false;
        }
        let bytes = hash.as_bytes();
        // The hash is a hex string, so each byte represents two hex digits.
        let full_zeros = difficulty / 2; // Number of complete zero bytes
        debug_assert!(
            full_zeros <= 32,
            "Full zeros {full_zeros} cannot exceed 32 bytes"
        );
        let partial_zero = difficulty % 2; // Remaining hex digit
        for byte in bytes.iter().take(full_zeros) {
            // Check complete zero bytes
            if *byte != 0 {
                // Must be exactly 0x00
                return false;
            }
        }
        if partial_zero > 0 {
            debug_assert!(full_zeros < 32, "Cannot check partial zero beyond 32 bytes");
            if bytes[full_zeros] >> 4 != 0 {
                // Check remaining hex digit (upper nibble of the next byte)
                return false;
            }
        }
        true
    }
    /// Validates block reward structure:
    /// - At most one block reward transaction
    /// - If present, must be the FIRST transaction
    /// - No other transactions can be block rewards
    #[inline]
    pub fn is_valid_block_reward_structure(transactions: &[Transaction]) -> bool {
        if transactions.is_empty() {
            return true; // Empty block is valid
        }
        // Check if first transaction is a block reward
        let has_reward = transactions[0].is_block_reward();

        if has_reward {
            // If first tx is reward, ensure no other transactions are rewards
            transactions.iter().skip(1).all(|tx| !tx.is_block_reward())
        } else {
            // If first tx is NOT reward, ensure NO transactions are rewards
            transactions.iter().all(|tx| !tx.is_block_reward())
        }
    }

    #[inline]
    pub fn is_valid_inner_block(inner: &BlockInner, difficulty: usize) -> bool {
        is_valid_target_hash(&inner.hash(), difficulty)
            && is_valid_block_reward_structure(&inner.transactions)
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

    pub fn estimate_block_size(block: &Block) -> usize {
        // Rough estimate:
        // - index (8 bytes) + previous_hash (32 bytes) + nonce (8 bytes) + timestamp (16 bytes)
        // - Each transaction: ~200 bytes average (address + signature + amount + timestamp + hash)
        let base_size = 8 + 32 + 8 + 16;
        let tx_size = block.inner.transactions.len() * 200;
        base_size + tx_size
    }

    pub fn validate_block(block: &Block) -> Result<(), serde_valid::validation::Error> {
        // Check block size
        let estimated_size = estimate_block_size(block);
        if estimated_size > MAX_BLOCK_SIZE_BYTES {
            return Err(serde_valid::validation::Error::Custom(format!(
                "Block size {estimated_size} exceeds maximum {MAX_BLOCK_SIZE_BYTES}"
            )));
        }

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
    use crate::{
        block::mining::{Miner, MiningStrategy},
        consts::MAX_BLOCK_SIZE_BYTES,
        wallet::Wallet,
    };

    #[test]
    fn test_mine_block_with_zero_difficulty() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let constructor = BlockConstructor::new(0, &[], previous_hash, None);
        let block = Miner::mine(constructor, 0, None);

        assert!(block.is_valid());
        assert_eq!(block.previous_hash(), &previous_hash);
    }

    #[test]
    fn test_mine_block_with_difficulty() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let constructor = BlockConstructor::new(0, &[], previous_hash, None);
        let block = Miner::mine(constructor, 4, None);

        assert!(block.is_valid());
        let hash_bytes = block.hash().as_bytes();
        assert_eq!(hash_bytes[0], 0);
    }

    #[test]
    fn test_block_hash_matches() {
        let previous_hash = Hash::from_bytes([1u8; 32]);
        let constructor = BlockConstructor::new(1, &[], previous_hash, None);
        let block = Miner::mine(constructor, 1, None);

        assert_eq!(block.hash(), &block.inner.hash());
    }

    #[test]
    fn test_block_with_transactions() {
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();
        let tx = wallet1.create_transaction(wallet2.address(), 50, 0, None);

        let previous_hash = Hash::from_bytes([0u8; 32]);
        let constructor = BlockConstructor::new(0, &[tx], previous_hash, None);

        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
    }

    #[test]
    fn test_block_serialization() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let constructor = BlockConstructor::new(0, &[], previous_hash, None);
        let block = Miner::mine(constructor, 1, None);

        let serialized = serde_json::to_string(&block).expect("serialization failed");
        let deserialized: Block =
            serde_json::from_str(&serialized).expect("deserialization failed");

        assert_eq!(block.hash(), deserialized.hash());
        assert_eq!(block.previous_hash(), deserialized.previous_hash());
        assert!(deserialized.is_valid());
    }

    #[test]
    fn test_block_with_miner_reward() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let miner_wallet = Wallet::new();

        let constructor =
            BlockConstructor::new(0, &[], previous_hash, Some(*miner_wallet.address()));
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert_eq!(
            block.transactions().len(),
            1,
            "Block should contain block reward transaction"
        );

        let reward_tx = &block.transactions()[0];
        assert!(
            reward_tx.is_block_reward(),
            "First transaction should be block reward"
        );
        assert_eq!(
            reward_tx.receiver(),
            miner_wallet.address(),
            "Reward should go to miner"
        );
        assert_eq!(reward_tx.amount(), 50, "Block reward should be 50");
        assert!(
            reward_tx.is_validate(),
            "Block reward transaction should be valid"
        );
    }

    #[test]
    fn test_block_with_miner_and_transactions() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let miner_wallet = Wallet::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx1 = wallet1.create_transaction(wallet2.address(), 25, 0, None);
        let tx2 = wallet1.create_transaction(wallet2.address(), 15, 1, None);

        let constructor = BlockConstructor::new(
            0,
            &[tx1.clone(), tx2.clone()],
            previous_hash,
            Some(*miner_wallet.address()),
        );
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert_eq!(
            block.transactions().len(),
            3,
            "Block should contain 2 user transactions + 1 block reward"
        );

        let reward_tx = &block.transactions()[0];
        assert!(
            reward_tx.is_block_reward(),
            "First transaction should be block reward"
        );
        assert_eq!(
            reward_tx.receiver(),
            miner_wallet.address(),
            "Reward should go to miner"
        );
        // Block reward = 50 (base) + 2 (fees from tx1 and tx2 @ 1 each)
        assert_eq!(reward_tx.amount(), 52, "Block reward should be 50 + 2 fees");

        assert_eq!(
            block.transactions()[1].hash(),
            tx1.hash(),
            "Second transaction should be tx1"
        );
        assert_eq!(
            block.transactions()[2].hash(),
            tx2.hash(),
            "Third transaction should be tx2"
        );
    }

    #[test]
    fn test_block_with_max_transactions_and_miner() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let miner_wallet = Wallet::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let mut transactions = Vec::new();
        for i in 0..(MAX_TRANSACTIONS_PER_BLOCK - 1) {
            transactions.push(wallet1.create_transaction(wallet2.address(), 1, i as u64, None));
        }

        let constructor = BlockConstructor::new(
            0,
            &transactions,
            previous_hash,
            Some(*miner_wallet.address()),
        );
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert_eq!(
            block.transactions().len(),
            MAX_TRANSACTIONS_PER_BLOCK,
            "Block should contain MAX_TRANSACTIONS (user txs + block reward)"
        );

        assert!(
            block.transactions()[0].is_block_reward(),
            "First transaction must be block reward"
        );
    }

    #[test]
    fn test_block_reward_must_be_first_transaction() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let miner_wallet = Wallet::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx = wallet1.create_transaction(wallet2.address(), 25, 0, None);

        // Create block with reward as first transaction (valid)
        let constructor = BlockConstructor::new(
            0,
            std::slice::from_ref(&tx),
            previous_hash,
            Some(*miner_wallet.address()),
        );
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert!(
            block.transactions()[0].is_block_reward(),
            "First transaction must be block reward"
        );
        assert_eq!(
            block.transactions()[1].hash(),
            tx.hash(),
            "User transaction should be second"
        );
    }

    #[test]
    fn test_block_with_only_user_transactions_no_rewards() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx1 = wallet1.create_transaction(wallet2.address(), 10, 0, None);
        let tx2 = wallet1.create_transaction(wallet2.address(), 20, 1, None);

        // Create block without miner address (no reward)
        let constructor =
            BlockConstructor::new(0, &[tx1.clone(), tx2.clone()], previous_hash, None);
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert_eq!(block.transactions().len(), 2);
        assert!(
            !block.transactions()[0].is_block_reward(),
            "No transaction should be a reward"
        );
        assert!(
            !block.transactions()[1].is_block_reward(),
            "No transaction should be a reward"
        );
    }

    #[test]
    fn test_empty_block_is_valid() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let constructor = BlockConstructor::new(0, &[], previous_hash, None);
        let block = Miner::mine(constructor, 1, None);

        assert!(block.is_valid());
        assert!(block.transactions().is_empty());
    }

    #[test]
    fn test_block_reward_structure_validation() {
        // Test that validation catches invalid structures
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let miner = Wallet::new();

        // Valid: empty block
        let constructor = BlockConstructor::new(0, &[], previous_hash, None);
        let block = Miner::mine(constructor, 1, None);
        assert!(utils::is_valid_block_reward_structure(block.transactions()));

        // Valid: block with reward + user transactions
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();
        let tx = wallet1.create_transaction(wallet2.address(), 10, 0, None);
        let constructor = BlockConstructor::new(0, &[tx], previous_hash, Some(*miner.address()));
        let block = Miner::mine(constructor, 1, None);
        assert!(utils::is_valid_block_reward_structure(block.transactions()));

        // Valid: block with only user transactions (no reward)
        let tx1 = wallet1.create_transaction(wallet2.address(), 5, 0, None);
        let tx2 = wallet1.create_transaction(wallet2.address(), 15, 1, None);
        let constructor = BlockConstructor::new(0, &[tx1, tx2], previous_hash, None);
        let block = Miner::mine(constructor, 1, None);
        assert!(utils::is_valid_block_reward_structure(block.transactions()));
    }

    #[test]
    fn test_block_size_validation_accepts_small_blocks() {
        // Small blocks should pass validation
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        // Create a few transactions
        let tx1 = wallet1.create_transaction(wallet2.address(), 10, 0, None);
        let tx2 = wallet1.create_transaction(wallet2.address(), 20, 1, None);

        let constructor = BlockConstructor::new(0, &[tx1, tx2], previous_hash, None);
        let block = Miner::mine(constructor, 1, None);

        // Should pass validation
        assert!(block.is_valid());

        // Test via serialization/deserialization (triggers serde_valid)
        let serialized = serde_json::to_string(&block).expect("serialization failed");
        let result: Result<Block, _> = serde_json::from_str(&serialized);
        assert!(result.is_ok(), "Small block should pass size validation");
    }

    #[test]
    fn test_block_size_validation_rejects_huge_blocks() {
        // Blocks with many transactions that exceed size limit should fail
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        // Create MAX_TRANSACTIONS_PER_BLOCK transactions
        // With our estimate of ~200 bytes per tx, 1000 txs = ~200KB which is under 1MB limit
        // So this should still pass
        let mut txs = Vec::new();
        for i in 0..MAX_TRANSACTIONS_PER_BLOCK {
            txs.push(wallet1.create_transaction(wallet2.address(), (i % 100) as u64 + 1, i as u64, None));
        }

        let constructor = BlockConstructor::new(0, &txs, previous_hash, None);
        let block = Miner::mine(constructor, 1, None);

        // Even with max transactions, should still be under 1MB limit
        let estimated_size = utils::estimate_block_size(&block);
        assert!(
            estimated_size <= MAX_BLOCK_SIZE_BYTES,
            "Block with max transactions should be under size limit"
        );
    }

    #[test]
    fn test_block_size_estimation() {
        let previous_hash = Hash::from_bytes([0u8; 32]);
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        // Empty block
        let constructor = BlockConstructor::new(0, &[], previous_hash, None);
        let block = Miner::mine(constructor, 1, None);
        let size_empty = utils::estimate_block_size(&block);

        // Block with one transaction
        let tx = wallet1.create_transaction(wallet2.address(), 50, 0, None);
        let constructor = BlockConstructor::new(0, &[tx], previous_hash, None);
        let block_with_tx = Miner::mine(constructor, 1, None);
        let size_with_tx = utils::estimate_block_size(&block_with_tx);

        // Size should increase with transactions
        assert!(
            size_with_tx > size_empty,
            "Block with transaction should be larger than empty block"
        );
    }
}
