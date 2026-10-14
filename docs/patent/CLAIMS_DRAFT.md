# Draft claims (for review by a registered patent agent)

*Working draft, not legal advice. Ten claims, so no excess-claim fee. Every
element maps to code, a test and a measured result in
[CLAIM_EVIDENCE_MATRIX.md](CLAIM_EVIDENCE_MATRIX.md). Reference numerals follow
[REFERENCE_NUMERALS.md](REFERENCE_NUMERALS.md). There is deliberately no
computer-readable-medium claim: under the CRI Guidelines 2025 a weak claim of one
category can sink the subject matter of all of them.*

*The claims follow Indian practice: "We claim:", each claim one sentence,
dependent claims referring back by number. Prior art to distinguish: Toyota
US 12,258,045 B2 (sensor-quality metadata for electing training vehicles),
US 2022/0414464 A1 (data-quality-weighted aggregation), Bonawitz et al. CCS 2017
(pairwise-mask secure aggregation), Sahu et al. NeurIPS 2021 (budgeted top-k),
arXiv 2401.07224 (dwell-time vehicle selection). None of these, as far as the
snippet-level search could tell, aggregates quality weights that the server
never sees, or allocates per-vehicle prefixes of one shared support by
predicted airtime under secure aggregation. A full-text search by the agent is
still required.*

---

**We claim:**

1. A method of training a shared machine-learning model across a plurality of
vehicles (100, 102) by a road-side unit (200) over a vehicle-to-infrastructure
radio link (130), the method comprising, for a training round, by the road-side
unit:
(a) selecting a set of the vehicles to take part in the round;
(b) for each selected vehicle, predicting an uplink rate of the radio link from a
reported position of the vehicle, and assigning to the vehicle a prefix length
from a predetermined ladder of prefix lengths (802), the assigned prefix length
being the longest prefix length whose transmission time at the predicted uplink
rate fits within an equal share of an uplink airtime budget for the round;
(c) transmitting to the selected vehicles the update of the global model made in
the preceding round, the assigned prefix lengths, and a public key of each
selected vehicle;
(d) receiving from selected vehicles masked update messages (700), each message
consisting of ring elements of a ring of integers modulo 2^b covering the
vehicle's prefix of an ordered coordinate support (800) and, for each segment
(801) of the support covered by that prefix, a masked fixed-point encoding of an
aggregation weight private to the vehicle and of the product of that weight and
the number of local training steps of the vehicle;
(e) summing the received messages in the ring, whereby pairwise masks added by
the vehicles cancel, to obtain for each segment a sum of weighted step-normalised
updates, a sum of the weights and a sum of the weight-step products, without
obtaining the weight, the step count or the update of any individual vehicle;
(f) computing for each segment an aggregate update equal to the ratio of the sum
of weight-step products to the sum of weights, multiplied by the sum of weighted
updates divided by the sum of weights; and
(g) updating the global model with the aggregate update;
wherein the ordered coordinate support is derived identically by the road-side
unit and by each vehicle from the magnitudes of the coordinates of the update of
the global model made in the preceding round, so that no coordinate index is
transmitted on the radio link.

2. A method performed by an on-board unit (110) of a vehicle (100) taking part in
a training round of a shared machine-learning model coordinated by a road-side
unit (200), the method comprising:
(a) receiving from the road-side unit the update of the global model made in the
preceding round, an assigned prefix length, and public keys of the other vehicles
selected for the round;
(b) training a local copy of the model on sensor data of the vehicle stored in
the on-board unit, over a number of local steps, to obtain a local update;
(c) computing a quality score of the vehicle from an output of a sensor-health
and calibration monitor (112) of the vehicle and from a similarity between a
label distribution of the sensor data and a fleet reference distribution, and an
aggregation weight from the quality score and the amount of sensor data;
(d) dividing the local update by the number of local steps and adding an
error-feedback residual held in a memory (115) of the on-board unit, to obtain a
corrected update;
(e) deriving the ordered coordinate support from the magnitudes of the
coordinates of the received global model update, and taking the corrected update
on the prefix of the support of the assigned prefix length;
(f) encoding the product of the aggregation weight and the corrected update on
the prefix as fixed-point elements of a ring of integers modulo 2^b after
clipping to a bound, and returning the clipped excess to the residual;
(g) appending, for each segment of the support covered by the prefix, a
fixed-point encoding of the aggregation weight and of the product of the
aggregation weight and the number of local steps;
(h) adding, for each other selected vehicle, a pseudo-random mask derived from a
Diffie-Hellman shared secret between a private key held in a secure element
(117) of the vehicle and the public key of the other vehicle, over only those
coordinates and segments that both vehicles transmit, with opposite signs for
the two vehicles of each pair;
(i) transmitting the resulting masked update message (700) with a checksum over
the radio link; and
(j) when the road-side unit acknowledges receipt, retaining in the residual only
the corrected update outside the transmitted prefix together with the clipped
excess, and when the road-side unit signals a failed reception or no
acknowledgement is received within the round, retaining the whole corrected
update in the residual.

3. A system for training a shared machine-learning model across vehicles,
comprising a road-side unit (200) having a V2I transceiver (201), a link-rate
estimator (202), a tier scheduler (204), a secure aggregation unit (205), a
processor (208) and a memory (209) configured to perform the method of claim 1,
and a plurality of vehicles (100, 102) each having an on-board unit (110) with a
sensor interface (111), a sensor-health and calibration monitor (112), a
processor with a neural accelerator (113), a memory (114) holding an
error-feedback residual (115), a secure element (117) holding a private key and a
V2X modem (118), the on-board unit being configured to perform the method of
claim 2.

4. The method of claim 1, further comprising, when a selected vehicle's message
is not received or fails its checksum, obtaining from each vehicle whose message
was received the pair seed it shares with that vehicle, and removing from the sum
the corresponding mask terms over the coordinates and segments common to the two
vehicles.

5. The method of claim 1, wherein the predetermined ladder of prefix lengths is a
geometric sequence between a minimum and a maximum length, and the maximum length
is adjusted from round to round by closed-loop control on the measured
communication volume of the preceding round against a per-round volume budget.

6. The method of claim 1, wherein the aggregation weight of each vehicle is capped
at a public maximum, and the road-side unit transmits to the vehicles a clipping
bound for the weighted update values derived from the largest magnitude of the
aggregate update of the preceding round and the public maximum weight.

7. The method of claim 1, wherein selecting the set of vehicles comprises
excluding vehicles whose predicted link uptime is below a threshold, stratifying
the remaining vehicles over road regions, and, before applying the threshold,
promoting any vehicle not selected within a predetermined number of rounds.

8. The method of claim 2, wherein the quality score is the product of a
sensor-health signal and one minus the Hellinger distance between the label
distribution and the fleet reference distribution, clamped to a range, and the
aggregation weight is the quality score multiplied by the ratio of the amount of
sensor data to a public reference amount.

9. The method of claim 2, wherein, in the first training round, the ordered
coordinate support is a pseudo-random permutation of the coordinates derived from
a seed known to the road-side unit and the vehicles.

10. The method of claim 2, further comprising clipping the local update to a norm
bound and adding Gaussian noise to it before step (d), the noise scale being
calibrated to a differential-privacy budget using the cap on the aggregation
weight as the bound on any one vehicle's influence on the aggregate.

---

### Notes for the agent

* Claim 1 combines K1 (quality-hiding weighted secure aggregation) with K2
  (airtime-budget tiers on one shared support). If a full-text search finds K2
  alone, step (b) and the segment structure can move to a dependent claim and
  claim 1 rests on K1 with the shared support.
* Claim 2 step (j) is the NACK-driven residual retention. It is measured in
  `results/patent/` as the difference between `proposed_tiered`
  (retention on) and `fedavg_transport` (retention off) under 2% corrupted
  uploads plus link drops.
* Claim 10 depends on the DP path; it is implemented (`fedav/client/trainer.py`,
  `fedav/privacy/dp.py`) but the multi-seed evidence was run with DP off. Run
  `configs/dp.yaml` before relying on it.
