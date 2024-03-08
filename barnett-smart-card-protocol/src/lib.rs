use crate::error::CardProtocolError;

use ark_ff::{Field, ToBytes};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::rand::Rng;
use proof_essentials::error::CryptoError;
use proof_essentials::homomorphic_encryption::HomomorphicEncryptionScheme;
use proof_essentials::utils::permutation::Permutation;
use proof_essentials::vector_commitment::HomomorphicCommitmentScheme;
use std::hash::Hash;
use std::ops::{Add, Mul};
use wasm_bindgen::prelude::*;


use anyhow;
use ark_ff::{to_bytes, UniformRand};
use ark_std::{One};
use proof_essentials::utils::rand::sample_vector;
use proof_essentials::zkp::proofs::{chaum_pedersen_dl_equality, schnorr_identification};
use rand::thread_rng;
use std::collections::HashMap;
use std::iter::Iterator;
use thiserror::Error;
use wasm_bindgen::prelude::*;

pub mod discrete_log_cards;
pub mod error;

#[wasm_bindgen]
pub fn test() -> String {
    "Hello, world!".to_string()
}
pub trait Mask<Scalar: Field, Enc: HomomorphicEncryptionScheme<Scalar>> {
    fn mask(
        &self,
        pp: &Enc::Parameters,
        shared_key: &Enc::PublicKey,
        r: &Scalar,
    ) -> Result<Enc::Ciphertext, CardProtocolError>;
}

pub trait Remask<Scalar: Field, Enc: HomomorphicEncryptionScheme<Scalar>> {
    fn remask(
        &self,
        pp: &Enc::Parameters,
        shared_key: &Enc::PublicKey,
        r: &Scalar,
    ) -> Result<Enc::Ciphertext, CardProtocolError>;
}

pub trait Reveal<F: Field, Enc: HomomorphicEncryptionScheme<F>> {
    fn reveal(&self, cipher: &Enc::Ciphertext) -> Result<Enc::Plaintext, CardProtocolError>;
}

/// Mental Poker protocol based on the one described by Barnett and Smart (2003).
/// The protocol has been modified to make use of the argument of a correct shuffle presented
/// by Bayer and Groth (2014).
pub trait BarnettSmartProtocol {
    // Cryptography
    type Scalar: Field;
    type Parameters;
    type PlayerPublicKey: CanonicalDeserialize + CanonicalSerialize;
    type PlayerSecretKey;
    type AggregatePublicKey: CanonicalDeserialize + CanonicalSerialize;
    type Enc: HomomorphicEncryptionScheme<Self::Scalar>;
    type Comm: HomomorphicCommitmentScheme<Self::Scalar>;

    // Cards
    type Card: Copy
        + Clone
        + Mask<Self::Scalar, Self::Enc>
        + CanonicalDeserialize
        + CanonicalSerialize
        + Hash
        + Eq;
    type MaskedCard: Remask<Self::Scalar, Self::Enc> + CanonicalDeserialize + CanonicalSerialize;
    type RevealToken: Add
        + Reveal<Self::Scalar, Self::Enc>
        + Mul<Self::Scalar, Output = Self::RevealToken>
        + CanonicalDeserialize
        + CanonicalSerialize;

    // Proofs
    type ZKProofKeyOwnership: CanonicalDeserialize + CanonicalSerialize;
    type ZKProofMasking: CanonicalDeserialize + CanonicalSerialize;
    type ZKProofRemasking: CanonicalDeserialize + CanonicalSerialize;
    type ZKProofReveal: CanonicalDeserialize + CanonicalSerialize;
    type ZKProofShuffle: CanonicalDeserialize + CanonicalSerialize;

    /// Randomly produce the scheme parameters
    fn setup<R: Rng>(
        rng: &mut R,
        m: usize,
        n: usize,
    ) -> Result<Self::Parameters, CardProtocolError>;

    /// Generate keys for a player.
    fn player_keygen<R: Rng>(
        rng: &mut R,
        pp: &Self::Parameters,
    ) -> Result<(Self::PlayerPublicKey, Self::PlayerSecretKey), CardProtocolError>;

    /// Prove in zero knowledge that the owner of a public key `pk` knows the corresponding secret key `sk`
    fn prove_key_ownership<B: ToBytes, R: Rng>(
        rng: &mut R,
        pp: &Self::Parameters,
        pk: &Self::PlayerPublicKey,
        sk: &Self::PlayerSecretKey,
        player_public_info: &B,
    ) -> Result<Self::ZKProofKeyOwnership, CryptoError>;

    /// Verify a proof od key ownership
    fn verify_key_ownership<B: ToBytes>(
        pp: &Self::Parameters,
        pk: &Self::PlayerPublicKey,
        player_public_info: &B,
        proof: &Self::ZKProofKeyOwnership,
    ) -> Result<(), CryptoError>;

    /// Use all the public keys and zk-proofs to compute a verified aggregate public key
    fn compute_aggregate_key<B: ToBytes>(
        pp: &Self::Parameters,
        player_keys_proof_info: &Vec<(Self::PlayerPublicKey, Self::ZKProofKeyOwnership, B)>,
    ) -> Result<Self::AggregatePublicKey, CardProtocolError>;

    /// Use the shared public key and a (private) random scalar `alpha` to mask a card.
    /// Returns a masked card and a zk-proof that the masking operation was applied correctly.
    fn mask<R: Rng>(
        rng: &mut R,
        pp: &Self::Parameters,
        shared_key: &Self::AggregatePublicKey,
        original_card: &Self::Card,
        alpha: &Self::Scalar,
    ) -> Result<(Self::MaskedCard, Self::ZKProofMasking), CardProtocolError>;

    /// Verify a proof of masking
    fn verify_mask(
        pp: &Self::Parameters,
        shared_key: &Self::AggregatePublicKey,
        card: &Self::Card,
        masked_card: &Self::MaskedCard,
        proof: &Self::ZKProofMasking,
    ) -> Result<(), CryptoError>;

    /// Use the shared public key and a (private) random scalar `alpha` to remask a masked card.
    /// Returns a masked card and a zk-proof that the remasking operation was applied correctly.
    fn remask<R: Rng>(
        rng: &mut R,
        pp: &Self::Parameters,
        shared_key: &Self::AggregatePublicKey,
        original_masked: &Self::MaskedCard,
        alpha: &Self::Scalar,
    ) -> Result<(Self::MaskedCard, Self::ZKProofRemasking), CardProtocolError>;

    /// Verify a proof of remasking
    fn verify_remask(
        pp: &Self::Parameters,
        shared_key: &Self::AggregatePublicKey,
        original_masked: &Self::MaskedCard,
        remasked: &Self::MaskedCard,
        proof: &Self::ZKProofRemasking,
    ) -> Result<(), CryptoError>;

    /// Players can use this function to compute their reveal token for a given masked card.
    /// The token is accompanied by a proof that it is a valid reveal for the specified card issued
    /// by the player who ran the computation.
    fn compute_reveal_token<R: Rng>(
        rng: &mut R,
        pp: &Self::Parameters,
        sk: &Self::PlayerSecretKey,
        pk: &Self::PlayerPublicKey,
        masked_card: &Self::MaskedCard,
    ) -> Result<(Self::RevealToken, Self::ZKProofReveal), CardProtocolError>;

    /// Verify a proof of correctly computed reveal token
    fn verify_reveal(
        pp: &Self::Parameters,
        pk: &Self::PlayerPublicKey,
        reveal_token: &Self::RevealToken,
        masked_card: &Self::MaskedCard,
        proof: &Self::ZKProofReveal,
    ) -> Result<(), CryptoError>;

    /// After collecting all the necessary reveal tokens and proofs that these are correctly issued,
    /// players can unmask a masked card to recover the underlying card.
    fn unmask(
        pp: &Self::Parameters,
        decryption_key: &Vec<(
            Self::RevealToken,
            Self::ZKProofReveal,
            Self::PlayerPublicKey,
        )>,
        masked_card: &Self::MaskedCard,
    ) -> Result<Self::Card, CardProtocolError>;

    /// Shuffle and remask a deck of masked cards using a player-chosen permutation and vector of
    /// masking factors.
    fn shuffle_and_remask<R: Rng>(
        rng: &mut R,
        pp: &Self::Parameters,
        shared_key: &Self::AggregatePublicKey,
        deck: &Vec<Self::MaskedCard>,
        masking_factors: &Vec<Self::Scalar>,
        permutation: &Permutation,
    ) -> Result<(Vec<Self::MaskedCard>, Self::ZKProofShuffle), CardProtocolError>;

    /// Verify a proof of correct shuffle
    fn verify_shuffle(
        pp: &Self::Parameters,
        shared_key: &Self::AggregatePublicKey,
        original_deck: &Vec<Self::MaskedCard>,
        shuffled_deck: &Vec<Self::MaskedCard>,
        proof: &Self::ZKProofShuffle,
    ) -> Result<(), CryptoError>;
}



// ################ EXAMPLE ROUND ################
// /example/rounds.rs copied here since wasm-pack works on /src/lib.rs

// Choose elliptic curve setting
type Curve = starknet_curve::Projective;
type Scalar = starknet_curve::Fr;

// Instantiate concrete type for our card protocol
type CardProtocol<'a> = discrete_log_cards::DLCards<'a, Curve>;
type CardParameters = discrete_log_cards::Parameters<Curve>;
type PublicKey = discrete_log_cards::PublicKey<Curve>;
type SecretKey = discrete_log_cards::PlayerSecretKey<Curve>;

type Card = discrete_log_cards::Card<Curve>;
type MaskedCard = discrete_log_cards::MaskedCard<Curve>;
type RevealToken = discrete_log_cards::RevealToken<Curve>;

type ProofKeyOwnership = schnorr_identification::proof::Proof<Curve>;
type RemaskingProof = chaum_pedersen_dl_equality::proof::Proof<Curve>;
type RevealProof = chaum_pedersen_dl_equality::proof::Proof<Curve>;

#[derive(Error, Debug, PartialEq)]
pub enum GameErrors {
    #[error("No such card in hand")]
    CardNotFound,

    #[error("Invalid card")]
    InvalidCard,
}

#[derive(PartialEq, Clone, Copy, Eq)]
pub enum Suite {
    Club,
    Diamond,
    Heart,
    Spade,
}

impl Suite {
    const VALUES: [Self; 4] = [Self::Club, Self::Diamond, Self::Heart, Self::Spade];
}

#[derive(PartialEq, PartialOrd, Clone, Copy, Eq)]
pub enum Value {
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
    Jack,
    Queen,
    King,
    Ace,
}

impl Value {
    const VALUES: [Self; 13] = [
        Self::Two,
        Self::Three,
        Self::Four,
        Self::Five,
        Self::Six,
        Self::Seven,
        Self::Eight,
        Self::Nine,
        Self::Ten,
        Self::Jack,
        Self::Queen,
        Self::King,
        Self::Ace,
    ];
}

#[derive(PartialEq, Clone, Eq, Copy)]
pub struct ClassicPlayingCard {
    value: Value,
    suite: Suite,
}

impl ClassicPlayingCard {
    pub fn new(value: Value, suite: Suite) -> Self {
        Self { value, suite }
    }
}

impl std::fmt::Debug for ClassicPlayingCard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let suite = match self.suite {
            Suite::Club => "♣",
            Suite::Diamond => "♦",
            Suite::Heart => "♥",
            Suite::Spade => "♠",
        };

        let val = match self.value {
            Value::Two => "2",
            Value::Three => "3",
            Value::Four => "4",
            Value::Five => "5",
            Value::Six => "6",
            Value::Seven => "7",
            Value::Eight => "8",
            Value::Nine => "9",
            Value::Ten => "10",
            Value::Jack => "J",
            Value::Queen => "Q",
            Value::King => "K",
            Value::Ace => "A",
        };

        write!(f, "{}{}", val, suite)
    }
}

#[derive(Clone)]
struct Player {
    name: Vec<u8>,
    sk: SecretKey,
    pk: PublicKey,
    proof_key: ProofKeyOwnership,
    cards: Vec<MaskedCard>,
    opened_cards: Vec<Option<ClassicPlayingCard>>,
}

impl Player {
    pub fn new<R: Rng>(rng: &mut R, pp: &CardParameters, name: &Vec<u8>) -> anyhow::Result<Self> {
        let (pk, sk) = CardProtocol::player_keygen(rng, pp)?;
        let proof_key = CardProtocol::prove_key_ownership(rng, pp, &pk, &sk, name)?;
        Ok(Self {
            name: name.clone(),
            sk,
            pk,
            proof_key,
            cards: vec![],
            opened_cards: vec![],
        })
    }

    pub fn receive_card(&mut self, card: MaskedCard) {
        self.cards.push(card);
        self.opened_cards.push(None);
    }

    pub fn peek_at_card(
        &mut self,
        parameters: &CardParameters,
        reveal_tokens: &mut Vec<(RevealToken, RevealProof, PublicKey)>,
        card_mappings: &HashMap<Card, ClassicPlayingCard>,
        card: &MaskedCard,
    ) -> Result<(), anyhow::Error> {
        let i = self.cards.iter().position(|&x| x == *card);

        let i = i.ok_or(GameErrors::CardNotFound)?;

        //TODO add function to create that without the proof
        let rng = &mut thread_rng();
        let own_reveal_token = self.compute_reveal_token(rng, parameters, card)?;
        reveal_tokens.push(own_reveal_token);

        let unmasked_card = CardProtocol::unmask(&parameters, reveal_tokens, card)?;
        let opened_card = card_mappings.get(&unmasked_card);
        let opened_card = opened_card.ok_or(GameErrors::InvalidCard)?;

        self.opened_cards[i] = Some(*opened_card);
        Ok(())
    }

    pub fn compute_reveal_token<R: Rng>(
        &self,
        rng: &mut R,
        pp: &CardParameters,
        card: &MaskedCard,
    ) -> anyhow::Result<(RevealToken, RevealProof, PublicKey)> {
        let (reveal_token, reveal_proof) =
            CardProtocol::compute_reveal_token(rng, &pp, &self.sk, &self.pk, card)?;

        Ok((reveal_token, reveal_proof, self.pk))
    }
}

//Every player will have to calculate this function for cards that are in play
pub fn open_card(
    parameters: &CardParameters,
    reveal_tokens: &Vec<(RevealToken, RevealProof, PublicKey)>,
    card_mappings: &HashMap<Card, ClassicPlayingCard>,
    card: &MaskedCard,
) -> Result<ClassicPlayingCard, anyhow::Error> {
    let unmasked_card = CardProtocol::unmask(&parameters, reveal_tokens, card)?;
    let opened_card = card_mappings.get(&unmasked_card);
    let opened_card = opened_card.ok_or(GameErrors::InvalidCard)?;

    Ok(*opened_card)
}

fn encode_cards<R: Rng>(rng: &mut R, num_of_cards: usize) -> HashMap<Card, ClassicPlayingCard> {
    let mut map: HashMap<Card, ClassicPlayingCard> = HashMap::new();
    let plaintexts = (0..num_of_cards)
        .map(|_| Card::rand(rng))
        .collect::<Vec<_>>();

    let mut i = 0;
    for value in Value::VALUES.iter().copied() {
        for suite in Suite::VALUES.iter().copied() {
            let current_card = ClassicPlayingCard::new(value, suite);
            map.insert(plaintexts[i], current_card);
            i += 1;
        }
    }

    map
}



#[wasm_bindgen]
pub fn protocl() -> anyhow::Result<String,JsValue> {
    let m = 2;
    let n = 26;
    let num_of_cards = m * n;
    let rng = &mut thread_rng();

    let parameters = CardProtocol::setup(rng, m, n).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let card_mapping = encode_cards(rng, num_of_cards);

    let mut andrija = Player::new(rng, &parameters, &to_bytes![b"Andrija"].unwrap()).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut kobi = Player::new(rng, &parameters, &to_bytes![b"Kobi"].unwrap()).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut nico = Player::new(rng, &parameters, &to_bytes![b"Nico"].unwrap()).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut tom = Player::new(rng, &parameters, &to_bytes![b"Tom"].unwrap()).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let players = vec![andrija.clone(), kobi.clone(), nico.clone(), tom.clone()];

    let key_proof_info = players
        .iter()
        .map(|p| (p.pk, p.proof_key, p.name.clone()))
        .collect::<Vec<_>>();

    // Each player should run this computation. Alternatively, it can be ran by a smart contract
    let joint_pk = CardProtocol::compute_aggregate_key(&parameters, &key_proof_info).map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Each player should run this computation and verify that all players agree on the initial deck
    let deck_and_proofs: Vec<(MaskedCard, RemaskingProof)> = card_mapping
        .keys()
        .map(|card| CardProtocol::mask(rng, &parameters, &joint_pk, &card, &Scalar::one()))
        .collect::<Result<Vec<_>, _>>().map_err(|e| JsValue::from_str(&e.to_string()))?;

    let deck = deck_and_proofs
        .iter()
        .map(|x| x.0)
        .collect::<Vec<MaskedCard>>();

    // SHUFFLE TIME --------------
    // 1.a Andrija shuffles first
    let permutation = Permutation::new(rng, m * n);
    let masking_factors: Vec<Scalar> = sample_vector(rng, m * n);

    let (a_shuffled_deck, a_shuffle_proof) = CardProtocol::shuffle_and_remask(
        rng,
        &parameters,
        &joint_pk,
        &deck,
        &masking_factors,
        &permutation,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;

    // 1.b everyone checks!
    CardProtocol::verify_shuffle(
        &parameters,
        &joint_pk,
        &deck,
        &a_shuffled_deck,
        &a_shuffle_proof,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;

    //2.a Kobi shuffles second
    let permutation = Permutation::new(rng, m * n);
    let masking_factors: Vec<Scalar> = sample_vector(rng, m * n);

    let (k_shuffled_deck, k_shuffle_proof) = CardProtocol::shuffle_and_remask(
        rng,
        &parameters,
        &joint_pk,
        &a_shuffled_deck,
        &masking_factors,
        &permutation,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;

    //2.b Everyone checks
    CardProtocol::verify_shuffle(
        &parameters,
        &joint_pk,
        &a_shuffled_deck,
        &k_shuffled_deck,
        &k_shuffle_proof,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;

    //3.a Nico shuffles third
    let permutation = Permutation::new(rng, m * n);
    let masking_factors: Vec<Scalar> = sample_vector(rng, m * n);

    let (n_shuffled_deck, n_shuffle_proof) = CardProtocol::shuffle_and_remask(
        rng,
        &parameters,
        &joint_pk,
        &k_shuffled_deck,
        &masking_factors,
        &permutation,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;

    //3.b Everyone checks
    CardProtocol::verify_shuffle(
        &parameters,
        &joint_pk,
        &k_shuffled_deck,
        &n_shuffled_deck,
        &n_shuffle_proof,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;

    //4.a Tom shuffles last
    let permutation = Permutation::new(rng, m * n);
    let masking_factors: Vec<Scalar> = sample_vector(rng, m * n);

    let (final_shuffled_deck, final_shuffle_proof) = CardProtocol::shuffle_and_remask(
        rng,
        &parameters,
        &joint_pk,
        &n_shuffled_deck,
        &masking_factors,
        &permutation,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;

    //4.b Everyone checks before accepting last deck for game
    CardProtocol::verify_shuffle(
        &parameters,
        &joint_pk,
        &n_shuffled_deck,
        &final_shuffled_deck,
        &final_shuffle_proof,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;

    // CARDS ARE SHUFFLED. ROUND OF THE GAME CAN BEGIN
    let deck = final_shuffled_deck;

    andrija.receive_card(deck[0]);
    kobi.receive_card(deck[1]);
    nico.receive_card(deck[2]);
    tom.receive_card(deck[3]);

    let andrija_rt_1 = andrija.compute_reveal_token(rng, &parameters, &deck[1]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let andrija_rt_2 = andrija.compute_reveal_token(rng, &parameters, &deck[2]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let andrija_rt_3 = andrija.compute_reveal_token(rng, &parameters, &deck[3]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let kobi_rt_0 = kobi.compute_reveal_token(rng, &parameters, &deck[0]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let kobi_rt_2 = kobi.compute_reveal_token(rng, &parameters, &deck[2]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let kobi_rt_3 = kobi.compute_reveal_token(rng, &parameters, &deck[3]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let nico_rt_0 = nico.compute_reveal_token(rng, &parameters, &deck[0]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let nico_rt_1 = nico.compute_reveal_token(rng, &parameters, &deck[1]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let nico_rt_3 = nico.compute_reveal_token(rng, &parameters, &deck[3]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let tom_rt_0 = tom.compute_reveal_token(rng, &parameters, &deck[0]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let tom_rt_1 = tom.compute_reveal_token(rng, &parameters, &deck[1]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let tom_rt_2 = tom.compute_reveal_token(rng, &parameters, &deck[2]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut rts_andrija = vec![kobi_rt_0, nico_rt_0, tom_rt_0];
    let mut rts_kobi = vec![andrija_rt_1, nico_rt_1, tom_rt_1];
    let mut rts_nico = vec![andrija_rt_2, kobi_rt_2, tom_rt_2];
    let mut rts_tom = vec![andrija_rt_3, kobi_rt_3, nico_rt_3];

    //At this moment players privately open their cards and only they know that values
    andrija.peek_at_card(&parameters, &mut rts_andrija, &card_mapping, &deck[0]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    kobi.peek_at_card(&parameters, &mut rts_kobi, &card_mapping, &deck[1]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    nico.peek_at_card(&parameters, &mut rts_nico, &card_mapping, &deck[2]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    tom.peek_at_card(&parameters, &mut rts_tom, &card_mapping, &deck[3]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    /* Here we can add custom logic of a game:
        1. swap card
        2. place a bet
        3. ...
    */

    //At this moment players reveal their cards to each other and everything becomes public

    //1.a everyone reveals the secret for their card
    let andrija_rt_0 = andrija.compute_reveal_token(rng, &parameters, &deck[0]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let kobi_rt_1 = kobi.compute_reveal_token(rng, &parameters, &deck[1]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let nico_rt_2 = nico.compute_reveal_token(rng, &parameters, &deck[2]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let tom_rt_3 = tom.compute_reveal_token(rng, &parameters, &deck[3]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    //2. tokens for all other cards are exchanged
    //TODO add struct for this so that we can just clone
    let andrija_rt_1 = andrija.compute_reveal_token(rng, &parameters, &deck[1]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let andrija_rt_2 = andrija.compute_reveal_token(rng, &parameters, &deck[2]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let andrija_rt_3 = andrija.compute_reveal_token(rng, &parameters, &deck[3]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let kobi_rt_0 = kobi.compute_reveal_token(rng, &parameters, &deck[0]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let kobi_rt_2 = kobi.compute_reveal_token(rng, &parameters, &deck[2]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let kobi_rt_3 = kobi.compute_reveal_token(rng, &parameters, &deck[3]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let nico_rt_0 = nico.compute_reveal_token(rng, &parameters, &deck[0]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let nico_rt_1 = nico.compute_reveal_token(rng, &parameters, &deck[1]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let nico_rt_3 = nico.compute_reveal_token(rng, &parameters, &deck[3]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let tom_rt_0 = tom.compute_reveal_token(rng, &parameters, &deck[0]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let tom_rt_1 = tom.compute_reveal_token(rng, &parameters, &deck[1]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let tom_rt_2 = tom.compute_reveal_token(rng, &parameters, &deck[2]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    let rt_0 = vec![andrija_rt_0, kobi_rt_0, nico_rt_0, tom_rt_0];
    let rt_1 = vec![andrija_rt_1, kobi_rt_1, nico_rt_1, tom_rt_1];
    let rt_2 = vec![andrija_rt_2, kobi_rt_2, nico_rt_2, tom_rt_2];
    let rt_3 = vec![andrija_rt_3, kobi_rt_3, nico_rt_3, tom_rt_3];

    //Everyone computes for each card (except for their own card):
    let andrija_card = open_card(&parameters, &rt_0, &card_mapping, &deck[0]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let kobi_card = open_card(&parameters, &rt_1, &card_mapping, &deck[1]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let nico_card = open_card(&parameters, &rt_2, &card_mapping, &deck[2]).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let tom_card = open_card(&parameters, &rt_3, &card_mapping, &deck[3]).map_err(|e| JsValue::from_str(&e.to_string()))?;

    println!("Andrija: {:?}", andrija_card);
    println!("Kobi: {:?}", kobi_card);
    println!("Nico: {:?}", nico_card);
    println!("Tom: {:?}", tom_card);

    let mut str = String::new();
    str.push_str(&format!("Andrija: {:?}\n", andrija_card));
    str.push_str(&format!("Kobi: {:?}\n", kobi_card));
    str.push_str(&format!("Nico: {:?}\n", nico_card));
    str.push_str(&format!("Tom: {:?}\n", tom_card));

    
    Ok(str)
}
