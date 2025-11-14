use std::collections::{HashSet, VecDeque};

use blake3::Hash;
use blockchain_types::{Transaction, consts::TRANSACTION_EXPIRY_SECONDS};
use chrono::Utc;

#[derive(Debug, Clone)]
pub struct Mempool {
    transactions: VecDeque<Transaction>,
    seen_hashes: HashSet<Hash>,
}

impl Mempool {
    pub fn new() -> Self {
        Self {
            transactions: VecDeque::new(),
            seen_hashes: HashSet::new(),
        }
    }

    /// Add transaction if not already seen
    pub fn add_transaction(&mut self, tx: Transaction) -> bool {
        let tx_hash = tx.hash();
        if self.seen_hashes.insert(tx_hash) {
            self.transactions.push_back(tx);
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

    /// Get the next transaction from the mempool
    pub fn get_transaction(&mut self) -> Option<Transaction> {
        let tx = self.transactions.pop_front();
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
        self.transactions
            .retain(|tx| !tx_hashes.contains(&tx.hash()));
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

        self.transactions.retain(|tx| {
            let age_seconds = now.signed_duration_since(tx.timestamp()).num_seconds();
            let is_expired = age_seconds > TRANSACTION_EXPIRY_SECONDS;

            // Remove hash from seen_hashes if expired
            if is_expired {
                self.seen_hashes.remove(&tx.hash());
            }

            !is_expired
        });

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

        let tx = wallet1.create_transaction(wallet2.address(), 50);

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

        let tx1 = wallet1.create_transaction(wallet2.address(), 50);
        let tx2 = wallet1.create_transaction(wallet2.address(), 30);

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

        let tx1 = wallet1.create_transaction(wallet2.address(), 50);
        let tx2 = wallet1.create_transaction(wallet2.address(), 30);
        let tx3 = wallet1.create_transaction(wallet2.address(), 20);

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

        let tx = wallet1.create_transaction(wallet2.address(), 50);

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

        let tx = wallet1.create_transaction(wallet2.address(), 50);

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

        let tx1 = wallet1.create_transaction(wallet2.address(), 50);
        let tx2 = wallet1.create_transaction(wallet2.address(), 30);

        // Add both transactions
        assert!(mempool.add_transaction(tx1.clone()));
        assert!(mempool.add_transaction(tx2.clone()));
        assert_eq!(mempool.len(), 2);

        // Remove first transaction
        mempool.remove_transactions(&[tx1.clone()]);
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
        let tx = wallet1.create_transaction(wallet2.address(), 50);
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

        let tx = wallet1.create_transaction(wallet2.address(), 50);
        assert!(mempool.add_transaction(tx));

        // Prune (won't remove anything as it's fresh)
        mempool.prune_expired();

        // Verify consistency maintained
        assert_eq!(mempool.len(), mempool.seen_hashes.len());
    }
}
