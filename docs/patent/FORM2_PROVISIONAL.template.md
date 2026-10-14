# FORM 2 — THE PATENTS ACT, 1970 (39 of 1970) & THE PATENTS RULES, 2003

*PROVISIONAL SPECIFICATION (see section 10; rule 13). Draft for review by a
registered patent agent; not legal advice. Written as fully as a complete
specification, because a claim not fairly based on the provisional takes the
later date. Applicant and inventor fields must be completed by the people
concerned (see INVENTION_DISCLOSURE.md). Check `python
scripts/check_spec_limits.py` before filing.*

**Applicant:** [NAME, NATIONALITY, ADDRESS]

## TITLE

QUALITY-HIDING, AIRTIME-TIERED SECURE AGGREGATION FOR FEDERATED TRAINING OF VEHICLE PERCEPTION MODELS OVER VEHICLE-TO-INFRASTRUCTURE LINKS

## PREAMBLE

The following specification describes the invention.

## FIELD OF THE INVENTION

The invention relates to vehicle-to-infrastructure (V2I) communication and to
distributed training of machine-learning models in connected vehicles. More
particularly it relates to a road-side unit and to on-board units of vehicles
that jointly train a shared perception and trajectory model by exchanging
masked model updates over a shared radio uplink, so that the airtime each
vehicle occupies is bounded and the road-side unit learns neither any
individual vehicle's update nor its quality-based aggregation weight.

## BACKGROUND OF THE INVENTION

Connected vehicles record camera, LiDAR and radar data that could improve the
perception models running in them, but moving that data off the vehicles is
costly over radio links and exposes private information about drivers and
places. Federated learning (McMahan et al., 2017) trains a shared model by
having each vehicle compute a model update on its own data and sending only the
update to an aggregator, which forms a weighted average.

Applied over a V2I link this leaves three problems unsolved. First, all vehicles
in a road-side unit's cell share one uplink; a vehicle at the cell edge or out
of line of sight needs several times longer to send the same update, so one slow
vehicle holds the round open. Known compression (half-precision encoding,
top-k sparsification with error feedback, e.g. Sahu et al., 2021) shrinks the
bytes but does not bound the time on the channel. Second, vehicles' updates are
not equally useful: a de-calibrated camera or a vehicle that only sees one road
type contributes a biased update. Weighting updates by a data-quality score is
known (e.g. US 2022/0414464 A1), as is using sensor-quality metadata to decide
which vehicles train (US 12,258,045 B2). In those schemes the aggregator
receives each vehicle's quality score in the clear, although a score computed
from sensor health and label distribution is itself information about that
vehicle. Third, secure aggregation with pairwise cancelling masks (Bonawitz et
al., 2017) hides individual updates, but masking makes every transmitted
coordinate dense, so it cannot be combined with per-vehicle sparsification
without sending the whole vector; and an upload lost on the mobile link both
breaks mask cancellation and, with error feedback, loses the coordinates that
had already been removed from the vehicle's residual.

## OBJECTS OF THE INVENTION

An object of the invention is to bound the uplink airtime of a federated
training round on a shared V2I channel while keeping every vehicle's
contribution.

Another object is to weight vehicles' updates by a quality score without the
road-side unit learning any individual score, update, data amount or step count.

Another object is to combine secure aggregation with sparsified updates so that
no coordinate indices are transmitted and masks still cancel exactly.

A further object is to lose no part of an update when an upload is lost or
corrupted.

## SUMMARY OF THE INVENTION

The road-side unit (200) predicts each selected vehicle's uplink rate from its
reported position and assigns it the longest prefix, from a public ladder of
prefix lengths (802), that fits an equal share of a per-round uplink airtime
budget. All parties derive the same ordered coordinate support (800) from the
magnitudes of the previous round's global model update, so a vehicle's prefix is
a prefix of the same list and no indices are sent. Each vehicle encodes, as
fixed-point elements of the ring of integers modulo 2^32, its quality-weighted,
step-normalised update on its prefix and, for each covered segment (801), its
weight and the product of its weight and local step count; it adds pairwise
masks derived from Diffie-Hellman shared secrets over the coordinates it shares
with each other vehicle. Summing in the ring cancels the masks exactly and
yields per segment the sums of weighted updates, weights and weight-step
products, from which the road-side unit computes a quality-weighted,
step-normalised (FedNova) update. A failed upload triggers retention of the
whole update in the vehicle's error-feedback residual, and the other vehicles
reveal their pair seeds with the missing vehicle so its masks can be removed.

## BRIEF DESCRIPTION OF THE DRAWINGS

FIG. 1 shows the system: vehicles (100, 102) with on-board units (110) in the
coverage cell (210) of a road-side unit (200) connected by a backhaul (300) to a
model repository (310).

FIG. 2 is a block diagram of the on-board unit (110).

FIG. 3 is a block diagram of the road-side unit (200).

FIG. 4 is a flow diagram of the method in the on-board unit (steps S401–S410).

FIG. 5 is a flow diagram of the method in the road-side unit (steps S501–S510).

FIG. 6 is a sequence diagram of one training round with a lost upload.

FIG. 7 shows the masked update message (700).

FIG. 8 shows the ordered shared support (800), its segments (801) and the tier
prefix lengths (802).

FIG. 9 shows measured traffic, uplink airtime and accuracy of the invention
against two baselines.

## DETAILED DESCRIPTION OF THE INVENTION

### System (FIG. 1 to FIG. 3)

A fleet of vehicles (100, 102) moves through the coverage cell (210) of a
road-side unit (200). Each vehicle has an on-board unit (110) comprising a
sensor interface (111) receiving data from sensors (120), a sensor-health and
calibration monitor (112), a processor with a neural accelerator (113), a memory
(114) holding a local training data store (116) and an error-feedback residual
buffer (115), a secure element (117) holding a Diffie-Hellman private key, and a
V2X modem (118) with antenna (119) operating in the 5.9 GHz ITS band. The
road-side unit (200) comprises a V2I transceiver (201), a link-rate estimator
(202), a vehicle selector (203), a tier scheduler (204), a secure aggregation
unit (205), a processor (208), a memory (209) holding the global model (206),
and a backhaul interface (207). Only masked update messages (700) and control
messages cross the radio link (130); raw sensor data stays in the on-board unit.

### Method at the road-side unit (FIG. 5)

S501. The vehicle selector (203) selects a fraction of the vehicles in the cell.
A vehicle is eligible when its predicted link uptime is at least a threshold
(0.55 in the best method), the threshold being relaxed in steps of ×0.7 if too
few qualify. Before the threshold is applied, any vehicle not selected within K
rounds (K = 12) is promoted, so that vehicles on weak links are not starved. The
remaining places are filled round-robin over road regions, by a score favouring
stable links, healthy sensors and less frequently selected vehicles.

S502–S503. The link-rate estimator (202) converts each selected vehicle's
reported position into a predicted uplink rate using the mean path loss of the
3GPP TR 38.901 urban-micro model used for V2I links in TR 37.885, without the
shadowing it cannot know in advance. The tier scheduler (204) holds a public
ladder of prefix lengths L_1 < … < L_m, geometric between a minimum (2% of the
model parameters) and a maximum (the current top-k ratio times the number of
parameters, the ratio being adjusted ×0.7 or ×1.15 each round to keep traffic
under a per-round volume budget). Each vehicle is assigned the largest tier t
for which (L_t + 2(t+1)) × 4 bytes fits in an equal share of the round's uplink
airtime budget at its predicted rate.

S504. The road-side unit multicasts the previous round's global model update,
the assigned tiers, the public keys of the selected vehicles and a clipping
bound B for weighted update values, B being four times the maximum weight times
the largest magnitude of the previous aggregate update.

S505–S507. The road-side unit receives masked messages (700) and checks their
CRC-32. For every selected vehicle whose message is missing or corrupt, each
vehicle whose message arrived reveals the 8-byte pair seed it shares with the
missing vehicle, and the secure aggregation unit (205) subtracts the
corresponding mask over the coordinates and segments common to the two vehicles.

S508–S509. The received vectors are summed modulo 2^32. For each segment t of
the support, the sums S_t of weighted step-normalised updates, W_t of weights
and T_t of weight-step products are decoded from fixed point, and the segment's
aggregate update is d_t = (T_t / W_t) × S_t / W_t. The road-side unit never
obtains a quantity of a single vehicle.

S510. The aggregate is scattered onto the support coordinates, server momentum
(β = 0.9) is applied and the global model is updated; the update just made
defines the next round's support.

### Method at the on-board unit (FIG. 4)

S401–S402. The on-board unit receives the global model update, its tier and the
peer public keys, and trains its copy of the model for E local epochs (E = 2 in
the best method) by SGD on data in the store (116), taking τ_k steps.

S403. The sensor-health monitor (112) yields a sensor-health signal; the
on-board unit computes the quality score q_k as that signal times one minus the
Hellinger distance between the label distribution of its data and a fleet
reference distribution, clamped to [0.05, 1], and the weight
w_k = q_k × n_k / n_ref, capped at 4, n_k being its number of training samples
and n_ref a public reference amount.

S404–S405. It divides its update by τ_k, adds the residual (115), derives the
ordered support (the coordinates of the received global update sorted by
magnitude; in round 1 a permutation seeded by a value known to all parties) and
takes the values on the prefix of its tier.

S406. It encodes w_k times those values as signed fixed point in the ring
Z_2^32 with six bits of headroom for up to 64 summands, after clipping to ±B;
the clipped excess and the rounding error return to the residual. For each
segment its prefix covers it appends the fixed-point encodings of w_k and
w_k τ_k.

S407. For each other selected vehicle j it computes the pair seed as the first
8 bytes of SHA-256(g^{ab} mod p ‖ round), using the RFC 3526 2048-bit MODP group,
expands it into uniform 32-bit words, and adds them (if its identifier is lower)
or subtracts them (otherwise) on the coordinates and segments that both vehicles
send, i.e. up to the smaller of the two tiers.

S408–S410. It transmits the message (700): magic (701), vehicle identifier
(702), round (703), tier (704), value count (705), CRC-32 (706), masked values
(707) and masked segment scalars (708) (FIG. 7). On acknowledgement, the residual
keeps the corrected update outside the sent prefix plus the clipped excess; on a
negative acknowledgement or timeout it keeps the whole corrected update.

### Best method and working example

The best method known to the applicant uses the parameters stated above with a
multitask convolutional network of 0.70 M parameters (24-channel stem,
192-dimensional embedding, heads for 6-class classification, box regression and
a 6-step trajectory), SGD with learning rate 0.01, cosine decay, momentum 0.9,
weight decay 5×10⁻⁴, batch 32, a 9 MB per-round traffic budget and a 12 s
per-round uplink airtime budget, as set out in configs/patent_eval.yaml of the
accompanying implementation. The radio parameters are 5.9 GHz, 10 MHz,
23 dBm, noise figure 9 dB, 75% of Shannon capacity capped at 5.55 bit/s/Hz.

In a working example, 50 simulated vehicles with label skew (Dirichlet 0.4),
quantity skew, region- and weather-dependent image statistics and 12%
de-calibrated sensors trained for 30 rounds, 15 vehicles per round, with 2% of
uploads corrupted in addition to link drops. Each of three random seeds ran
four arms on the same fleet: (A) standard federated averaging with
full-precision uploads, (B) federated averaging on the same compression
transport as the invention (half precision, top-k with error feedback,
multicast, volume budget), (C) the invention without airtime tiers, and (D) the
invention with airtime tiers. The results (mean ± 95% confidence interval over
seeds) are:

[[RESULTS TABLE]]

In every round of arms C and D the aggregate recovered by the road-side unit
matched the plaintext quality-weighted, step-normalised aggregate to a relative
error of [[ERR]], the residue of fixed-point rounding; masks cancelled exactly.

## CLAIMS

*Claims are not required in a provisional specification; the draft claims in
CLAIMS_DRAFT.md are reproduced here so that the provisional fairly supports
them.*

[[CLAIMS]]

## ABSTRACT

Quality-hiding, airtime-tiered secure aggregation for federated training of vehicle perception models over vehicle-to-infrastructure links. A road-side unit (200) assigns each vehicle (100) a prefix of an ordered coordinate support (800), derived by all parties from the previous global update, sized to fit an equal share of an uplink airtime budget at the vehicle's predicted rate. Each vehicle encodes its quality-weighted, step-normalised update on that prefix, with its weight and weight-step product per segment, as integers modulo 2^32, adds pairwise Diffie-Hellman masks over coordinates shared with each peer, and sends the message (700). Summing cancels the masks, giving the road-side unit only sums from which it forms a quality-weighted update, never an individual vehicle's weight or update. Failed uploads keep the whole update in the vehicle's residual; survivors reveal pair seeds so masks cancel. FIG. 1.
