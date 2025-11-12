use blake3::{Hash, Hasher};
use chrono::{DateTime, Utc};
use ed25519_dalek::Signature;
use serde::{Deserialize, Serialize};
use serde_valid::Validate;

use crate::{
    consts::BLOCK_REWARD_SIGNATURE_BYTES,
    wallet::{Address, private_key::PrivateKey},
};

pub struct TransactionConstructor;

impl TransactionConstructor {
    pub fn new_transaction(
        sender: &Address,
        receiver: &Address,
        amount: u64,
        sender_private_key: &PrivateKey,
    ) -> Transaction {
        debug_assert!(amount > 0, "Transaction amount must be greater than 0");
        debug_assert_ne!(
            sender.as_bytes(),
            receiver.as_bytes(),
            "Sender and receiver must be different"
        );
        let inner = InnerTransaction::new(*sender, *receiver, amount);
        // 1. hash the transaction data
        let message_hash = inner.hash();
        // 2. sign the hash with sender's private key
        let signature = sender_private_key.sign(message_hash.as_bytes());
        // 3. compute the final transaction hash including the signature
        let transaction_hash = utils::hash_transaction_complete(Hasher::new(), &inner, signature);
        Transaction::new(inner, signature, transaction_hash)
    }

    pub(super) fn block_reward(miner_address: &Address, amount: u64) -> Transaction {
        debug_assert!(amount > 0, "Block reward amount must be greater than 0");
        let inner = InnerTransaction::block_reward(*miner_address, amount);
        // Block rewards use a dummy signature (all zeros)
        let dummy_signature = Signature::from_bytes(&BLOCK_REWARD_SIGNATURE_BYTES);
        // Hash includes the dummy signature
        let transaction_hash =
            utils::hash_transaction_complete(Hasher::new(), &inner, dummy_signature);
        Transaction::new_block_reward(inner, dummy_signature, transaction_hash)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TransactionType {
    Transfer { sender: Address, receiver: Address },
    BlockReward { miner: Address },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct InnerTransaction {
    tx_type: TransactionType,
    amount: u64,
    timestamp: DateTime<Utc>,
}

impl InnerTransaction {
    pub fn hash(&self) -> Hash {
        utils::hash_inner_transaction(Hasher::new(), self)
    }

    fn new(sender: Address, receiver: Address, amount: u64) -> Self {
        Self {
            tx_type: TransactionType::Transfer { sender, receiver },
            amount,
            timestamp: Utc::now(),
        }
    }

    fn block_reward(miner: Address, amount: u64) -> Self {
        Self {
            tx_type: TransactionType::BlockReward { miner },
            amount,
            timestamp: Utc::now(),
        }
    }

    fn is_block_reward(&self) -> bool {
        matches!(self.tx_type, TransactionType::BlockReward { .. })
    }
}

/// A valid transaction with signature and hash
/// Can only be created via a `Wallet` or `TransactionConstructor`
#[derive(Debug, Clone, Serialize, Deserialize, Validate, PartialEq, Eq)]
#[validate(custom = utils::validate_transaction)] // to uphold our invariant that a Transaction is always valid (even when deserialized)
pub struct Transaction {
    #[serde(flatten)]
    inner: InnerTransaction,
    signature: Signature,
    hash: Hash,
}

impl Transaction {
    pub fn hash(&self) -> Hash {
        self.hash
    }

    fn new(inner: InnerTransaction, signature: Signature, hash: Hash) -> Self {
        assert!(
            utils::is_valid_transaction(&inner, &signature, &hash),
            "Transaction must be valid: signature and hash must match inner transaction"
        );
        Self {
            inner,
            signature,
            hash,
        }
    }

    /// Create a block reward transaction (internal use only)
    /// Block rewards don't require signature validation since they create new coins
    fn new_block_reward(inner: InnerTransaction, signature: Signature, hash: Hash) -> Self {
        assert!(
            utils::is_valid_block_reward(&inner, &signature, &hash),
            "Block reward must be valid: must use genesis sender and correct hash"
        );
        Self {
            inner,
            signature,
            hash,
        }
    }

    // Checks if a transaction is valid (verify signature and hash NOT amount checks)
    pub fn is_validate(&self) -> bool {
        utils::is_valid_transaction(&self.inner, &self.signature, &self.hash)
    }

    // Add public accessors
    pub fn sender(&self) -> Option<&Address> {
        match &self.inner.tx_type {
            TransactionType::Transfer { sender, .. } => Some(sender),
            TransactionType::BlockReward { .. } => None, // No sender!
        }
    }

    pub fn receiver(&self) -> &Address {
        match &self.inner.tx_type {
            TransactionType::Transfer { receiver, .. } => receiver,
            TransactionType::BlockReward { miner } => miner,
        }
    }

    pub fn amount(&self) -> u64 {
        debug_assert!(self.inner.amount > 0, "Transaction amount must be > 0");
        self.inner.amount
    }

    pub fn timestamp(&self) -> DateTime<Utc> {
        self.inner.timestamp
    }

    pub fn is_block_reward(&self) -> bool {
        matches!(self.inner.tx_type, TransactionType::BlockReward { .. })
    }
}

// Ignore block reward for now
// enum TransactionType {
//     Transfer,
//     // BlockReward,
// }

mod utils {
    use super::{
        DateTime, Hash, Hasher, InnerTransaction, Signature, Transaction, TransactionType, Utc,
    };
    use crate::consts::BLOCK_REWARD_SIGNATURE_BYTES;

    #[inline]
    fn hash_transaction_type(hasher: &mut Hasher, tx_type: &TransactionType) {
        match tx_type {
            TransactionType::Transfer { sender, receiver } => {
                hasher.update(b"transfer"); // Type discriminator
                hasher.update(sender.as_bytes());
                hasher.update(receiver.as_bytes());
            }
            TransactionType::BlockReward { miner } => {
                hasher.update(b"block_reward"); // Type discriminator
                hasher.update(miner.as_bytes());
            }
        }
    }

    #[inline]
    fn hash_transaction_inner_data(
        mut hasher: Hasher,
        tx_type: &TransactionType,
        amount: u64,
        timestamp: DateTime<Utc>,
    ) -> Hasher {
        hash_transaction_type(&mut hasher, tx_type);
        hasher
            .update(&amount.to_le_bytes())
            .update(&timestamp.timestamp_millis().to_le_bytes());
        hasher
    }
    #[inline]
    fn hash_transaction_data(
        hasher: Hasher,
        tx_type: &TransactionType,
        amount: u64,
        timestamp: DateTime<Utc>,
    ) -> Hash {
        hash_transaction_inner_data(hasher, tx_type, amount, timestamp).finalize()
    }

    #[inline]
    pub fn hash_inner_transaction(hasher: Hasher, inner_tx: &InnerTransaction) -> Hash {
        hash_transaction_data(
            hasher,
            &inner_tx.tx_type,
            inner_tx.amount,
            inner_tx.timestamp,
        )
    }

    #[inline]
    fn hash_transaction_inner(
        hasher: Hasher,
        tx_type: &TransactionType,
        amount: u64,
        timestamp: DateTime<Utc>,
        signature: Signature,
    ) -> Hasher {
        let mut hasher = hash_transaction_inner_data(hasher, tx_type, amount, timestamp);
        hasher.update(signature.to_bytes().as_ref());
        hasher
    }

    #[inline]
    fn hash_transaction(
        hasher: Hasher,
        tx_type: &TransactionType,
        amount: u64,
        timestamp: DateTime<Utc>,
        signature: Signature,
    ) -> Hash {
        hash_transaction_inner(hasher, tx_type, amount, timestamp, signature).finalize()
    }

    pub fn hash_transaction_complete(
        hasher: Hasher,
        inner_tx: &InnerTransaction,
        signature: Signature,
    ) -> Hash {
        hash_transaction(
            hasher,
            &inner_tx.tx_type,
            inner_tx.amount,
            inner_tx.timestamp,
            signature,
        )
    }
    #[inline]
    fn is_valid_signature(inner_tx: &InnerTransaction, signature: &Signature) -> bool {
        // Only transfer transactions need signature validation
        match &inner_tx.tx_type {
            TransactionType::Transfer { sender, .. } => {
                sender.verify(inner_tx.hash().as_bytes(), signature).is_ok()
            }
            TransactionType::BlockReward { .. } => {
                // Block rewards use dummy signature (all zeros)
                signature.to_bytes() == BLOCK_REWARD_SIGNATURE_BYTES
            }
        }
    }

    #[inline]
    fn is_valid_transaction_hash(
        hash: &Hash,
        inner: &InnerTransaction,
        signature: &Signature,
    ) -> bool {
        *hash == hash_transaction_complete(Hasher::new(), inner, *signature)
    }
    #[inline]
    pub fn is_valid_transaction(
        inner_tx: &InnerTransaction,
        signature: &Signature,
        hash: &Hash,
    ) -> bool {
        is_valid_signature(inner_tx, signature)
            && is_valid_transaction_hash(hash, inner_tx, signature)
    }

    #[inline]
    pub fn is_valid_block_reward(
        inner_tx: &InnerTransaction,
        signature: &Signature,
        hash: &Hash,
    ) -> bool {
        if !matches!(inner_tx.tx_type, TransactionType::BlockReward { .. }) {
            return false;
        }

        // Dummy signature validation
        if signature.to_bytes() != BLOCK_REWARD_SIGNATURE_BYTES {
            return false;
        }

        // Hash validation
        is_valid_transaction_hash(hash, inner_tx, signature)
    }

    pub fn validate_transaction(tx: &Transaction) -> Result<(), serde_valid::validation::Error> {
        if is_valid_transaction(&tx.inner, &tx.signature, &tx.hash) {
            Ok(())
        } else {
            Err(serde_valid::validation::Error::Custom(
                "Invalid transaction".to_owned(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    use super::*;

    #[test]
    fn test_create_valid_transaction() {
        let mut csprng = OsRng;
        let sk = SigningKey::generate(&mut csprng);
        let sender_pk = Address::from_verifying_key(sk.verifying_key());
        let sender_sk = PrivateKey::from_singing_key(sk);

        let receiver_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());

        let tx = TransactionConstructor::new_transaction(&sender_pk, &receiver_pk, 100, &sender_sk);

        assert_eq!(tx.sender(), Some(&sender_pk));
        assert_eq!(tx.receiver(), &receiver_pk);
        assert_eq!(tx.amount(), 100);
        assert!(tx.is_validate());
    }

    #[test]
    fn test_transaction_validation() {
        let mut csprng = OsRng;
        let sk = SigningKey::generate(&mut csprng);
        let sender_pk = Address::from_verifying_key(sk.verifying_key());
        let sender_sk = PrivateKey::from_singing_key(sk);

        let receiver_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());

        let tx = TransactionConstructor::new_transaction(&sender_pk, &receiver_pk, 50, &sender_sk);

        assert!(tx.is_validate());
    }

    #[test]
    fn test_transaction_hash_consistency() {
        let mut csprng = OsRng;
        let sk = SigningKey::generate(&mut csprng);
        let sender_pk = Address::from_verifying_key(sk.verifying_key());
        let sender_sk = PrivateKey::from_singing_key(sk);
        let receiver_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());

        let tx = TransactionConstructor::new_transaction(&sender_pk, &receiver_pk, 75, &sender_sk);
        let hash1 = tx.hash();
        let hash2 = tx.hash();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_transaction_serialization_roundtrip() {
        let mut csprng = OsRng;
        let sk = SigningKey::generate(&mut csprng);
        let sender_pk = Address::from_verifying_key(sk.verifying_key());
        let sender_sk = PrivateKey::from_singing_key(sk);
        let receiver_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());

        let tx = TransactionConstructor::new_transaction(&sender_pk, &receiver_pk, 123, &sender_sk);

        let serialized = serde_json::to_string(&tx).expect("serialization failed");
        let deserialized: Transaction =
            serde_json::from_str(&serialized).expect("deserialization failed");

        assert_eq!(tx.sender(), deserialized.sender());
        assert_eq!(tx.receiver(), deserialized.receiver());
        assert_eq!(tx.amount(), deserialized.amount());
        assert_eq!(tx.hash(), deserialized.hash());
        assert!(deserialized.is_validate());
    }

    #[test]
    fn test_block_reward_creation() {
        let miner_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());
        let reward = TransactionConstructor::block_reward(&miner_pk, 50);

        assert!(reward.is_block_reward());
        assert_eq!(reward.receiver(), &miner_pk);
        assert_eq!(reward.amount(), 50);
        assert_eq!(reward.sender(), None);
        assert!(reward.is_validate());
    }

    #[test]
    fn test_block_reward_serialization() {
        let miner_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());
        let reward = TransactionConstructor::block_reward(&miner_pk, 50);

        let serialized = serde_json::to_string(&reward).expect("serialization failed");
        let deserialized: Transaction =
            serde_json::from_str(&serialized).expect("deserialization failed");

        assert!(deserialized.is_block_reward());
        assert_eq!(reward.receiver(), deserialized.receiver());
        assert_eq!(reward.amount(), deserialized.amount());
        assert!(deserialized.is_validate());
    }

    #[test]
    fn test_block_reward_has_no_sender() {
        let miner_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());
        let reward = TransactionConstructor::block_reward(&miner_pk, 50);

        assert_eq!(reward.sender(), None, "Block rewards should have no sender");
        assert_eq!(
            reward.receiver(),
            &miner_pk,
            "Receiver should be miner address"
        );
    }

    #[test]
    fn test_transfer_has_sender() {
        let mut csprng = OsRng;
        let sk = SigningKey::generate(&mut csprng);
        let sender_pk = Address::from_verifying_key(sk.verifying_key());
        let sender_sk = PrivateKey::from_singing_key(sk);
        let receiver_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());

        let tx = TransactionConstructor::new_transaction(&sender_pk, &receiver_pk, 100, &sender_sk);

        assert_eq!(tx.sender(), Some(&sender_pk), "Transfer should have sender");
        assert_eq!(tx.receiver(), &receiver_pk);
    }

    #[test]
    fn test_block_reward_uses_dummy_signature() {
        let miner_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());
        let reward = TransactionConstructor::block_reward(&miner_pk, 50);

        assert!(reward.is_validate(), "Block reward should be valid");
        assert!(
            reward.is_block_reward(),
            "Should be identified as block reward"
        );
    }

    #[test]
    fn test_transaction_type_discriminator_in_hash() {
        let miner_pk =
            Address::from_verifying_key(SigningKey::generate(&mut OsRng).verifying_key());

        let reward1 = TransactionConstructor::block_reward(&miner_pk, 50);
        std::thread::sleep(std::time::Duration::from_millis(5)); // Ensure different timestamp
        let reward2 = TransactionConstructor::block_reward(&miner_pk, 50);

        // Different timestamps mean different hashes even for same amount/receiver
        assert_ne!(
            reward1.hash(),
            reward2.hash(),
            "Different timestamps should produce different hashes"
        );

        // But both should be valid block rewards
        assert!(reward1.is_block_reward());
        assert!(reward2.is_block_reward());
    }
}
