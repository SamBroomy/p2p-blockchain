use std::collections::{BinaryHeap, HashSet};

use blake3::Hash;
use blockchain_types::{Transaction, consts::TRANSACTION_EXPIRY_SECONDS};
use chrono::Utc;

#[derive(Debug, Clone)]
pub struct Mempool {
    /// Priority queue ordered by transaction fee (highest first)
    transactions: BinaryHeap<Transaction>,
    seen_hashes: HashSet<Hash>,
}

impl Mempool {
    pub fn new() -> Self {
        Self {
            transactions: BinaryHeap::new(),
            seen_hashes: HashSet::new(),
        }
    }

    /// Add transaction if not already seen
    /// Transactions are automatically prioritized by fee (highest first)
    pub fn add_transaction(&mut self, tx: Transaction) -> bool {
        let tx_hash = tx.hash();
        if self.seen_hashes.insert(tx_hash) {
            self.transactions.push(tx);
            debug_assert_eq!(
                self.transactions.len(),
                self.seen_hashes.len(),
                "Transaction queue length {} must match seen hashes length {}",
                self.transactions.len(),
                self.seen_hashes.len()
            );
            true
        } else {
            false
        }
    }

    /// Get the highest-priority transaction from the mempool (highest fee first)
    pub fn get_transaction(&mut self) -> Option<Transaction> {
        let tx = self.transactions.pop();
        // Remove hash from seen_hashes to prevent memory leak and allow re-broadcast
        if let Some(ref transaction) = tx {
            self.seen_hashes.remove(&transaction.hash());
        }
        debug_assert_eq!(
            self.seen_hashes.len(),
            self.transactions.len(),
            "Seen hashes {} must equal transaction queue length {}",
            self.seen_hashes.len(),
            self.transactions.len()
        );
        tx
    }

    pub fn remove_transactions(&mut self, txs: &[Transaction]) {
        if txs.is_empty() {
            return;
        }
        let tx_hashes: HashSet<Hash> = txs.iter().map(Transaction::hash).collect();
        let old_len = self.transactions.len();

        // BinaryHeap doesn't have retain(), so drain, filter, and rebuild
        let filtered: Vec<Transaction> = self
            .transactions
            .drain()
            .filter(|tx| !tx_hashes.contains(&tx.hash()))
            .collect();
        self.transactions = BinaryHeap::from(filtered);

        // Remove hashes from seen_hashes to allow re-broadcast
        for hash in &tx_hashes {
            self.seen_hashes.remove(hash);
        }
        debug_assert!(
            old_len - self.transactions.len() <= txs.len(),
            "Removed {} transactions but only {} were provided",
            old_len - self.transactions.len(),
            txs.len()
        );
        debug_assert_eq!(
            self.seen_hashes.len(),
            self.transactions.len(),
            "Seen hashes must equal transaction queue length after removal"
        );
    }

    pub fn len(&self) -> usize {
        debug_assert!(
            self.transactions.len() <= self.seen_hashes.len(),
            "Transaction queue {} cannot exceed seen hashes {}",
            self.transactions.len(),
            self.seen_hashes.len()
        );
        self.transactions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }

    pub fn pending_transactions(&self) -> impl Iterator<Item = &Transaction> {
        self.transactions.iter()
    }

    /// Remove expired transactions (older than `TRANSACTION_EXPIRY_SECONDS`)
    pub fn prune_expired(&mut self) -> usize {
        let now = Utc::now();
        let initial_len = self.transactions.len();

        // BinaryHeap doesn't have retain(), so drain, filter, and rebuild
        let mut expired_hashes = Vec::new();
        let filtered: Vec<Transaction> = self
            .transactions
            .drain()
            .filter(|tx| {
                let age_seconds = now.signed_duration_since(tx.timestamp()).num_seconds();
                let is_expired = age_seconds > TRANSACTION_EXPIRY_SECONDS;

                if is_expired {
                    expired_hashes.push(tx.hash());
                }

                !is_expired
            })
            .collect();
        self.transactions = BinaryHeap::from(filtered);

        // Remove expired hashes from seen_hashes
        for hash in expired_hashes {
            self.seen_hashes.remove(&hash);
        }

        let removed_count = initial_len - self.transactions.len();
        debug_assert_eq!(
            self.seen_hashes.len(),
            self.transactions.len(),
            "Seen hashes must equal transaction queue length after pruning"
        );

        removed_count
    }
}
impl Default for Mempool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use blockchain_types::wallet::Wallet;

    use super::*;

    #[test]
    fn test_mempool_add_transaction() {
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx = wallet1.create_transaction(wallet2.address(), 50, 0, None);

        assert!(mempool.add_transaction(tx.clone()));
        assert_eq!(mempool.len(), 1);

        // Adding same transaction again should fail
        assert!(!mempool.add_transaction(tx));
        assert_eq!(mempool.len(), 1);
    }

    #[test]
    fn test_mempool_get_transaction() {
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx1 = wallet1.create_transaction(wallet2.address(), 50, 0, None);
        let tx2 = wallet1.create_transaction(wallet2.address(), 30, 0, None);

        mempool.add_transaction(tx1.clone());
        mempool.add_transaction(tx2.clone());

        assert_eq!(mempool.len(), 2);

        // FIFO order
        let retrieved = mempool.get_transaction().unwrap();
        assert_eq!(retrieved.hash(), tx1.hash());
        assert_eq!(mempool.len(), 1);
    }

    #[test]
    fn test_mempool_remove_transactions() {
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx1 = wallet1.create_transaction(wallet2.address(), 50, 0, None);
        let tx2 = wallet1.create_transaction(wallet2.address(), 30, 0, None);
        let tx3 = wallet1.create_transaction(wallet2.address(), 20, 0, None);

        mempool.add_transaction(tx1.clone());
        mempool.add_transaction(tx2.clone());
        mempool.add_transaction(tx3.clone());

        assert_eq!(mempool.len(), 3);

        mempool.remove_transactions(&[tx1, tx3]);
        assert_eq!(mempool.len(), 1);

        let remaining = mempool.get_transaction().unwrap();
        assert_eq!(remaining.hash(), tx2.hash());
    }

    #[test]
    fn test_mempool_deduplication() {
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx = wallet1.create_transaction(wallet2.address(), 50, 0, None);

        // First add succeeds
        assert!(mempool.add_transaction(tx.clone()));

        // Subsequent adds fail
        assert!(!mempool.add_transaction(tx.clone()));
        assert!(!mempool.add_transaction(tx));

        assert_eq!(mempool.len(), 1);
    }

    #[test]
    fn test_get_transaction_removes_hash() {
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx = wallet1.create_transaction(wallet2.address(), 50, 0, None);

        // Add transaction
        assert!(mempool.add_transaction(tx.clone()));
        assert_eq!(mempool.len(), 1);

        // Get transaction (removes from queue and hash set)
        let retrieved = mempool.get_transaction().unwrap();
        assert_eq!(retrieved.hash(), tx.hash());
        assert_eq!(mempool.len(), 0);

        // Should be able to re-add the same transaction now
        assert!(
            mempool.add_transaction(tx),
            "Transaction should be re-addable after being retrieved"
        );
        assert_eq!(mempool.len(), 1);
    }

    #[test]
    fn test_remove_transactions_removes_hashes() {
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx1 = wallet1.create_transaction(wallet2.address(), 50, 0, None);
        let tx2 = wallet1.create_transaction(wallet2.address(), 30, 0, None);

        // Add both transactions
        assert!(mempool.add_transaction(tx1.clone()));
        assert!(mempool.add_transaction(tx2.clone()));
        assert_eq!(mempool.len(), 2);

        // Remove first transaction
        mempool.remove_transactions(std::slice::from_ref(&tx1));
        assert_eq!(mempool.len(), 1);

        // Should be able to re-add the removed transaction
        assert!(
            mempool.add_transaction(tx1),
            "Removed transaction should be re-addable"
        );
        assert_eq!(mempool.len(), 2);
    }

    #[test]
    fn test_prune_expired_transactions() {
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        // Add a transaction
        let tx = wallet1.create_transaction(wallet2.address(), 50, 0, None);
        assert!(mempool.add_transaction(tx.clone()));
        assert_eq!(mempool.len(), 1);

        // Immediately pruning should not remove it (not expired)
        let removed = mempool.prune_expired();
        assert_eq!(removed, 0);
        assert_eq!(mempool.len(), 1);

        // Note: We can't easily test actual expiration in a unit test without
        // waiting 24 hours or mocking time. This test just verifies the method works.
    }

    #[test]
    fn test_prune_expired_removes_hashes() {
        // This test verifies that prune_expired properly maintains the seen_hashes set
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        let tx = wallet1.create_transaction(wallet2.address(), 50, 0, None);
        assert!(mempool.add_transaction(tx));

        // Prune (won't remove anything as it's fresh)
        mempool.prune_expired();

        // Verify consistency maintained
        assert_eq!(mempool.len(), mempool.seen_hashes.len());
    }

    #[test]
    fn test_fee_based_priority() {
        use blockchain_types::transaction::BlockFee;

        // Test that transactions are retrieved in fee priority order
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        // Add transactions with different fees (out of priority order)
        let tx_low = wallet1.create_transaction(wallet2.address(), 10, 0, BlockFee::Low);
        let tx_high = wallet1.create_transaction(wallet2.address(), 20, 1, BlockFee::High);
        let tx_medium = wallet1.create_transaction(wallet2.address(), 15, 2, BlockFee::Medium);

        mempool.add_transaction(tx_low.clone());
        mempool.add_transaction(tx_high.clone());
        mempool.add_transaction(tx_medium.clone());

        assert_eq!(mempool.len(), 3);

        // Should retrieve in fee priority order: High (10), Medium (5), Low (1)
        let retrieved1 = mempool.get_transaction().unwrap();
        assert_eq!(
            retrieved1.hash(),
            tx_high.hash(),
            "Should retrieve High fee transaction first"
        );
        assert_eq!(retrieved1.fee(), Some(10));

        let retrieved2 = mempool.get_transaction().unwrap();
        assert_eq!(
            retrieved2.hash(),
            tx_medium.hash(),
            "Should retrieve Medium fee transaction second"
        );
        assert_eq!(retrieved2.fee(), Some(5));

        let retrieved3 = mempool.get_transaction().unwrap();
        assert_eq!(
            retrieved3.hash(),
            tx_low.hash(),
            "Should retrieve Low fee transaction last"
        );
        assert_eq!(retrieved3.fee(), Some(1));

        assert_eq!(mempool.len(), 0);
    }

    #[test]
    fn test_same_fee_ordering() {
        // Test that transactions with the same fee are ordered by timestamp (older first)
        let mut mempool = Mempool::new();
        let wallet1 = Wallet::new();
        let wallet2 = Wallet::new();

        // Create multiple transactions with the same fee (default Low)
        // They should be ordered by timestamp (which they are created in)
        let tx1 = wallet1.create_transaction(wallet2.address(), 10, 0, None);
        std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamps
        let tx2 = wallet1.create_transaction(wallet2.address(), 20, 1, None);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let tx3 = wallet1.create_transaction(wallet2.address(), 30, 2, None);

        // Add in reverse order
        mempool.add_transaction(tx3.clone());
        mempool.add_transaction(tx1.clone());
        mempool.add_transaction(tx2.clone());

        // Should retrieve in timestamp order (oldest first) since fees are equal
        let retrieved1 = mempool.get_transaction().unwrap();
        assert_eq!(
            retrieved1.hash(),
            tx1.hash(),
            "Should retrieve oldest transaction first"
        );

        let retrieved2 = mempool.get_transaction().unwrap();
        assert_eq!(
            retrieved2.hash(),
            tx2.hash(),
            "Should retrieve second oldest transaction"
        );

        let retrieved3 = mempool.get_transaction().unwrap();
        assert_eq!(
            retrieved3.hash(),
            tx3.hash(),
            "Should retrieve newest transaction last"
        );
    }
}
