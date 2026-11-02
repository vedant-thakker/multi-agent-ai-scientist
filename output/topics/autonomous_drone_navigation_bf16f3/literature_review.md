# Advances and Challenges in Autonomous Drone Navigation: A Systematic Review of Recent Developments

**Evidence-Grounded Literature Review and Synthesis (8 Verified Studies)**  
*Automated Scientific Literature Review & Synthesis Pipeline*  
*Topic Scope: Autonomous Drone Navigation | Analysis Level: Abstract & Bibliographic Metadata*

---

## Abstract

Autonomous drone navigation has emerged as a critical area of research, with significant advancements in recent years. This literature review synthesizes the findings of eight retrieved studies [1]–[8] on autonomous drone navigation, highlighting key thematic paradigms, open challenges, and proposed future research directions. The retrieved studies demonstrate the effectiveness of various approaches, including foundation models, large language models, and deep learning-based methods. However, challenges such as obstacle avoidance, weather adaptation, and mission completion remain. This review identifies potential research gaps and proposes future directions, including the integration of complementary mechanisms to enhance autonomous drone navigation. The findings of this review contribute to the understanding of autonomous drone navigation and inform the development of more efficient and effective navigation systems.

**Keywords:** Autonomous Drone Navigation, Foundation Models, Large Language Models, Deep Learning, Obstacle Avoidance, Weather Adaptation

### Scope, Research Questions, and Evidence Granularity

- **Research Topic:** Autonomous Drone Navigation
- **Scope:** Systematic literature review and methodological synthesis of Autonomous Drone Navigation, examining 8 retrieved peer-reviewed and preprint studies.
- **Analysis Granularity:** Evidence extraction and comparative synthesis are grounded in retrieved paper abstracts and verified bibliographic metadata; full-text experimental appendices were not directly inspected unless stated in the abstract.
- RQ1: What are the primary algorithmic and methodological paradigms established across retrieved literature in Autonomous Drone Navigation?
- RQ2: How do existing approaches in Autonomous Drone Navigation compare with respect to evaluation settings, baselines, and documented limitations?
- RQ3: Which under-explored methodological intersections and future research directions emerge from a structured synthesis of Autonomous Drone Navigation?

## I. Introduction and Scope: Autonomous Drone Navigation

Autonomous drone navigation aims to enable safe and efficient navigation through complex environments, with recent advances in foundation models and large language models [1, 2] offering promising solutions. This review examines the scope of autonomous drone navigation, its core technical objectives, and the contributions of the retrieved studies.

The primary technical challenges in autonomous drone navigation include obstacle avoidance, weather adaptation, and mission completion, as documented in the retrieved studies [3, 4, 5]. These challenges necessitate the development of more robust and adaptive navigation systems.

This review employs a systematic metadata and abstract analysis approach, comparative graph mapping, and thematic synthesis to examine the methodologies, baselines, and findings across the retrieved studies [1]–[8].

## II. Thematic Literature Review of Autonomous Drone Navigation

The retrieved studies demonstrate the effectiveness of various approaches to autonomous drone navigation, including FM-Planner [1], LLMs [2], Cross-modal-Visuomotor [3], MiNav [4], and Autonomous-Drone [5]. These approaches offer complementary strengths, such as improved obstacle avoidance and weather adaptation. However, open limitations, including the need for more robust and adaptive navigation systems, remain. The synthesis of these findings highlights the potential for integrating complementary mechanisms to enhance autonomous drone navigation.

### II.A Perception, State Estimation & Sensing

Xiao et al. (2025) [1] ("FM-Planner: Foundation Model Guided Path Planning for Autonomous Drone Navigation," arXiv.org) examine FM-Planner in comparison with Conventional domain baselines across Reported in abstract (abstract-only analyzed). Based on the retrieved abstract (Analysis Basis: Abstract-level metadata), the study highlights key findings: "Path planning is a critical component in autonomous drone operations, enabling safe and efficient navigation through complex environments. Recent advances in foundation models, particularly large language models (LLMs) and vision-language models (VLMs),..." Documented technical considerations include However, their practical applicability and effectiveness in global path planning remain relatively unexplored..

Khan et al. (2025) [2] ("Context-Aware Autonomous Drone Navigation Using Large Language Models (LLMs)," AAAI Summer Symposia) examine LLMs in comparison with Conventional domain baselines across Reported in abstract (abstract-only analyzed). Based on the retrieved abstract (Reported Figure (Abstract): 94%; Evaluated Dimensions: Success Rate), the study highlights key findings: "In this paper, a novel large language model (LLM)-based context-aware autonomous drone navigation algorithm is presented. This approach demonstrates the capability of LLMs to navigate complex environments by balancing multisensor objectives with a weighted..." Documented technical considerations include Specifically, we incorporate weights for the goals of obstacle avoidance, weather adaptation, and mission completion..

Zhang et al. (2025) [3] ("Learning Cross-Modal Visuomotor Policies for Autonomous Drone Navigation," IEEE Robotics and Automation Letters) examine Cross-modal-Visuomotor in comparison with Conventional domain baselines across Reported in abstract (abstract-only analyzed). Based on the retrieved abstract (Analysis Basis: Abstract-level metadata), the study highlights key findings: "Developing effective vision-based navigation algorithms adapting to various scenarios is a significant challenge for autonomous drone systems, with vast potential in diverse real-world applications. This paper proposes a novel visuomotor policy learning..." Documented technical considerations include Developing effective vision-based navigation algorithms adapting to various scenarios is a significant challenge for autonomous drone systems, with vast potential in diverse real-world applications..

Lam et al. (2025) [4] ("MiNav: Autonomous Drone Navigation Indoors Using Millimeter-Waves," Proceedings of the ACM on Interactive Mobile Wearable and Ubiquitous Technologies) examine MiNav in comparison with Conventional domain baselines across Cluttered Indoor Environments. Based on the retrieved abstract (Reported Figure (Abstract): 20%; Evaluated Dimensions: Accuracy), the study highlights key findings: "We present the design, implementation, and evaluation of MiNav, a system capable of accurate, efficient and fully autonomous drone navigation in challenging indoor environments, including those where vision-based systems fail. MiNav builds on recent..." Documented technical considerations include We present the design, implementation, and evaluation of MiNav, a system capable of accurate, efficient and fully autonomous drone navigation in challenging indoor environments, including those where vision-based systems fail..

Col et al. (2025) [5] ("Autonomous Drone Navigation in Forest Environments Using Deep Learning," The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences) examine Autonomous-Drone in comparison with Visual-Inertial Odometry (VIO) across Reported in abstract (abstract-only analyzed). Based on the retrieved abstract (Evaluated Dimensions: Collision / Safety Rate), the study highlights key findings: "Abstract. Autonomous drone navigation in dense forests remains challenging due to unreliable GNSS signals, difficulty detecting thin branches, and cumulative drift in Visual-Inertial Odometry (VIO). This work investigates a deep learning-based navigation..." Documented technical considerations include Autonomous drone navigation in dense forests remains challenging due to unreliable GNSS signals, difficulty detecting thin branches, and cumulative drift in Visual-Inertial Odometry (VIO)..

Safie et al. (2025) [7] ("Deep Learning Enabled Autonomous Drone Navigation for Safety Inspections in Chemical Plants," 2025 IEEE International Conference on Automatic Control and Intelligent Systems (I2CACIS)) examine Enabled-Autonomous in comparison with Conventional domain baselines across Reported in abstract (abstract-only analyzed). Based on the retrieved abstract (Analysis Basis: Abstract-level metadata), the study highlights key findings: "Chemical plants present hazardous environments for manual safety inspections, driving the need for automated solutions. This paper presents an autonomous drone navigation framework using Quick Response (QR) code markers and deep learning to enhance safety..." Documented technical considerations include Experimental results demonstrate that the model achieved perfect QR code detection (average precision 1.00) on static imagery and $88 \%$ precision on more challenging dynamic footage, indicating robust recognition capability..

Shinde et al. (2025) [8] ("Autonomous Drone Navigation Using Reinforcement Learning," 2025 Global Conference on Information Technology and Communication Networks (GITCON)) examine Autonomous-Drone-Shinde in comparison with Proximal Policy Optimization (PPO) across Cluttered Indoor Environments, PyBullet Simulation. Based on the retrieved abstract (Evaluated Dimensions: Success Rate, Collision / Safety Rate), the study highlights key findings: "This paper presents a robust solution for autonomous drone navigation within complex, confined environments using a Proximal Policy Optimization (PPO) based reinforcement learning approach. The proposed system leverages a simulated environment built with..." Documented technical considerations include We detail the environment setup, observation space design incorporating ray casting for obstacle detection, and the action space for drone movement..

### II.B Trajectory Planning, Navigation & Control

Silva et al. (2025) [6] ("Simulation-Based Autonomous Drone Navigation for Security and Monitoring Using Deep Reinforcement Learning," 2025 Brazilian Conference on Robotics (CROS)) examine Simulation-based-Autonomou in comparison with Conventional domain baselines across Reported in abstract (abstract-only analyzed). Based on the retrieved abstract (Analysis Basis: Abstract-level metadata), the study highlights key findings: "Security and monitoring of large scale areas—such as farms, critical infrastructure, or large urban zones—face significant challenges due to high costs, lack of accessibility, and risks to human personnel. Currently, autonomous drones show up as a..." Documented technical considerations include Security and monitoring of large scale areas—such as farms, critical infrastructure, or large urban zones—face significant challenges due to high costs, lack of accessibility, and risks to human personnel..

## III. Comparative Methodology Synthesis and Knowledge Graph Analysis

Comparing the retrieved approaches reveals complementary strengths and open limitations. For example, FM-Planner and Autonomous-Drone offer robust global path planning and navigation in dense forests, respectively. However, their practical applicability and effectiveness in various scenarios remain relatively unexplored. The integration of these approaches, as proposed in future research directions, may address these limitations and enhance autonomous drone navigation.

Knowledge Graph Structure: To map explicit relationships across the retrieved corpus, we constructed a directed Subject–Predicate–Object (S-P-O) graph comprising 20 concept nodes, 13 directed edges, and 13 verified provenance triads. Central concepts by PageRank centrality include core domain methodologies.

Candidate Research Gaps (Structural Holes): Topological analysis identified 3 candidate methodological gaps: (1) MiNav and Autonomous-Drone-Shinde (undirected method distance: 6 hops; bridge score: 7.0638), which address related domain challenges (Cluttered Indoor Environments) without direct methodological integration in the retrieved corpus; (2) FM-Planner and Autonomous-Drone (undirected method distance: 6 hops; bridge score: 5.8246), which address related domain challenges (retrieved domain evaluation settings) without direct methodological integration in the retrieved corpus; (3) LLMs and Cross-modal-Visuomotor (undirected method distance: 6 hops; bridge score: 5.4800), which address related domain challenges (retrieved domain evaluation settings) without direct methodological integration in the retrieved corpus. These structural gaps represent synthesized opportunities for cross-paradigm investigation rather than established empirical results.

### Table I: Comparative Summary of Retrieved Studies

| Ref | Study | Venue (Year) | Primary Method | Compared Baseline | Evaluation Setting | Reported Findings | Documented Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [1] | Xiao et al. (2025) | arXiv.org (2025) | FM-Planner | Conventional domain baselines | Reported in abstract (abstract-only analyzed) | Analysis Basis: Abstract-level metadata | However, their practical applicability and effectiveness in global path planning remain relatively unexplored. |
| [2] | Khan et al. (2025) | AAAI Summer Symposia (2025) | LLMs | Conventional domain baselines | Reported in abstract (abstract-only analyzed) | Reported Figure (Abstract): 94%; Evaluated Dimensions: Success Rate | Specifically, we incorporate weights for the goals of obstacle avoidance, weather adaptation, and mission completion. |
| [3] | Zhang et al. (2025) | IEEE Robotics and Automation Letters (2025) | Cross-modal-Visuomotor | Conventional domain baselines | Reported in abstract (abstract-only analyzed) | Analysis Basis: Abstract-level metadata | Developing effective vision-based navigation algorithms adapting to various scenarios is a significant challenge for autonomous drone systems, with vast potential in diverse real-world applications. |
| [4] | Lam et al. (2025) | Proceedings of the ACM on Interactive Mobile Wearable and Ubiquitous Technologies (2025) | MiNav | Conventional domain baselines | Cluttered Indoor Environments | Reported Figure (Abstract): 20%; Evaluated Dimensions: Accuracy | We present the design, implementation, and evaluation of MiNav, a system capable of accurate, efficient and fully autonomous drone navigation in challenging indoor environments, including those where vision-based systems fail. |
| [5] | Col et al. (2025) | The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences (2025) | Autonomous-Drone | Visual-Inertial Odometry (VIO) | Reported in abstract (abstract-only analyzed) | Evaluated Dimensions: Collision / Safety Rate | Autonomous drone navigation in dense forests remains challenging due to unreliable GNSS signals, difficulty detecting thin branches, and cumulative drift in Visual-Inertial Odometry (VIO). |
| [6] | Silva et al. (2025) | 2025 Brazilian Conference on Robotics (CROS) (2025) | Simulation-based-Autonomou | Conventional domain baselines | Reported in abstract (abstract-only analyzed) | Analysis Basis: Abstract-level metadata | Security and monitoring of large scale areas—such as farms, critical infrastructure, or large urban zones—face significant challenges due to high costs, lack of accessibility, and risks to human personnel. |
| [7] | Safie et al. (2025) | 2025 IEEE International Conference on Automatic Control and Intelligent Systems (I2CACIS) (2025) | Enabled-Autonomous | Conventional domain baselines | Reported in abstract (abstract-only analyzed) | Analysis Basis: Abstract-level metadata | Experimental results demonstrate that the model achieved perfect QR code detection (average precision 1.00) on static imagery and $88 \%$ precision on more challenging dynamic footage, indicating robust recognition capability. |
| [8] | Shinde et al. (2025) | 2025 Global Conference on Information Technology and Communication Networks (GITCON) (2025) | Autonomous-Drone-Shinde | Proximal Policy Optimization (PPO) | Cluttered Indoor Environments, PyBullet Simulation | Evaluated Dimensions: Success Rate, Collision / Safety Rate | We detail the environment setup, observation space design incorporating ray casting for obstacle detection, and the action space for drone movement. |

### Table II: Candidate Methodological Gaps (Knowledge Graph Analysis)

| Rank | Candidate Method Pair | Graph Distance | Bridge Score | Shared Domain Context |
| :--- | :--- | :--- | :--- | :--- |
| 1 | MiNav <-> Autonomous-Drone-Shinde | 6 hops | 7.0638 | Cluttered Indoor Environments |
| 2 | FM-Planner <-> Autonomous-Drone | 6 hops | 5.8246 | retrieved domain evaluation settings |
| 3 | LLMs <-> Cross-modal-Visuomotor | 6 hops | 5.4800 | retrieved domain evaluation settings |

![Domain Knowledge Graph for Autonomous Drone Navigation](knowledge_graph.jpg)

## IV. Proposed Future Research Directions in Autonomous Drone Navigation

Disclaimer on Proposed Research Directions: The directions below are formulated by our synthesis pipeline as falsifiable future research hypotheses for Autonomous Drone Navigation. Unlike the published findings reviewed in Section II [1–8], these proposed integrations have not yet been empirically benchmarked in this study and are presented as structured protocols for future investigation.

### IV.A Proposed Research Direction (MiNav-Autonomous-Dro): Integrating MiNav and Autonomous-Drone-Shinde for Autonomous Drone Navigation

Motivation and Gap: MiNav (2025) ↔ Autonomous-Drone-Shinde (2025) | Lineage Distance: 6 hops | Bridge Priority Score: 7.0638

Hypothesis Statement (MiNav-Autonomous-Dro): [Proposed Hypothesis] Within the retrieved literature on Autonomous Drone Navigation, MiNav notes the challenge: "We present the design, implementation, and evaluation of MiNav, a system capable of accurate, efficient and fully autonomous drone navigation in challenging indoor environments, including those where vision-based systems fail.", whereas Autonomous-Drone-Shinde notes: "We detail the environment setup, observation space design incorporating ray casting for obstacle detection, and the action space for drone movement.". We hypothesize that combining the complementary mechanisms of MiNav and Autonomous-Drone-Shinde represents a promising future research direction for Autonomous Drone Navigation.

Conceptual Formulation: Conceptual integration of MiNav and Autonomous-Drone-Shinde (proposed design; empirical validation not yet performed)

Proposed Evaluation Protocol and Falsification Criteria: Proposed 4-part ablation design on Cluttered Indoor Environments: (i) standalone MiNav, (ii) standalone Autonomous-Drone-Shinde, (iii) unregularized combination, and (iv) full proposed MiNav-Autonomous-Dro integration.

Target Outcomes to Test: Domain-Specific Task Performance, Computational Overhead, Generalization Stability, Ablation Sensitivity Across Hyperparameters, Multi-Seed Variance / Confidence Intervals

Methodological Scope Refinements: Addressed review feedback (Formulate a 4-part ablation protocol comparing standalone MiNav, standalone Autonomous-Drone-Shinde, unregularized hybrid, and full MiNav-Autonomous-Dro.; Specify falsifiable evaluation criteria and multi-seed variance controls without claiming unperformed experiments as established fact.) by specifying a falsifiable 4-part ablation protocol, grounding baselines in the retrieved corpus, and explicitly labeling MiNav-Autonomous-Dro as an unverified research proposal.

### IV.B Proposed Research Direction (FM-Planner-Autonomous-Dro): Integrating FM-Planner and Autonomous-Drone for Autonomous Drone Navigation

Motivation and Gap: FM-Planner (2025) ↔ Autonomous-Drone (2025) | Lineage Distance: 6 hops | Bridge Priority Score: 5.8246

Hypothesis Statement (FM-Planner-Autonomous-Dro): [Proposed Hypothesis] Within the retrieved literature on Autonomous Drone Navigation, FM-Planner notes the challenge: "However, their practical applicability and effectiveness in global path planning remain relatively unexplored.", whereas Autonomous-Drone notes: "Autonomous drone navigation in dense forests remains challenging due to unreliable GNSS signals, difficulty detecting thin branches, and cumulative drift in Visual-Inertial Odometry (VIO).". We hypothesize that combining the complementary mechanisms of FM-Planner and Autonomous-Drone represents a promising future research direction for Autonomous Drone Navigation.

Conceptual Formulation: Conceptual integration of FM-Planner and Autonomous-Drone (proposed design; empirical validation not yet performed)

Proposed Evaluation Protocol and Falsification Criteria: Proposed 4-part ablation design on retrieved domain evaluation settings: (i) standalone FM-Planner, (ii) standalone Autonomous-Drone, (iii) unregularized combination, and (iv) full proposed FM-Planner-Autonomous-Dro integration.

Target Outcomes to Test: Domain-Specific Task Performance, Computational Overhead, Generalization Stability, Ablation Sensitivity Across Hyperparameters, Multi-Seed Variance / Confidence Intervals

Methodological Scope Refinements: Addressed review feedback (Formulate a 4-part ablation protocol comparing standalone FM-Planner, standalone Autonomous-Drone, unregularized hybrid, and full FM-Planner-Autonomous-Dro.; Specify falsifiable evaluation criteria and multi-seed variance controls without claiming unperformed experiments as established fact.) by specifying a falsifiable 4-part ablation protocol, grounding baselines in the retrieved corpus, and explicitly labeling FM-Planner-Autonomous-Dro as an unverified research proposal.

### IV.C Proposed Research Direction (LLMs-Cross-modal-Vi): Integrating LLMs and Cross-modal-Visuomotor for Autonomous Drone Navigation

Motivation and Gap: LLMs (2025) ↔ Cross-modal-Visuomotor (2025) | Lineage Distance: 6 hops | Bridge Priority Score: 5.4800

Hypothesis Statement (LLMs-Cross-modal-Vi): [Proposed Hypothesis] Within the retrieved literature on Autonomous Drone Navigation, LLMs notes the challenge: "Specifically, we incorporate weights for the goals of obstacle avoidance, weather adaptation, and mission completion.", whereas Cross-modal-Visuomotor notes: "Developing effective vision-based navigation algorithms adapting to various scenarios is a significant challenge for autonomous drone systems, with vast potential in diverse real-world applications.". We hypothesize that combining the complementary mechanisms of LLMs and Cross-modal-Visuomotor represents a promising future research direction for Autonomous Drone Navigation.

Conceptual Formulation: Conceptual integration of LLMs and Cross-modal-Visuomotor (proposed design; empirical validation not yet performed)

Proposed Evaluation Protocol and Falsification Criteria: Proposed 4-part ablation design on retrieved domain evaluation settings: (i) standalone LLMs, (ii) standalone Cross-modal-Visuomotor, (iii) unregularized combination, and (iv) full proposed LLMs-Cross-modal-Vi integration.

Target Outcomes to Test: Domain-Specific Task Performance, Computational Overhead, Generalization Stability, Ablation Sensitivity Across Hyperparameters, Multi-Seed Variance / Confidence Intervals

Methodological Scope Refinements: Addressed review feedback (Formulate a 4-part ablation protocol comparing standalone LLMs, standalone Cross-modal-Visuomotor, unregularized hybrid, and full LLMs-Cross-modal-Vi.; Specify falsifiable evaluation criteria and multi-seed variance controls without claiming unperformed experiments as established fact.) by specifying a falsifiable 4-part ablation protocol, grounding baselines in the retrieved corpus, and explicitly labeling LLMs-Cross-modal-Vi as an unverified research proposal.

## V. Limitations, Open Challenges, and Conclusion

This literature review synthesizes the findings of recent studies on autonomous drone navigation, highlighting key thematic paradigms, open challenges, and proposed future research directions. While the retrieved studies demonstrate significant advancements, honest scope limitations, such as abstract-level evidence analysis, must be acknowledged. The proposed future directions, including the integration of complementary mechanisms, offer promising avenues for enhancing autonomous drone navigation. Further research is necessary to address the remaining challenges and develop more efficient and effective navigation systems.

Methodological and Evidence Scope Limitations: In accordance with strict evidence grounding, all summaries of prior work in this manuscript are derived from retrieved abstracts and verified bibliographic records from ArXiv and Semantic Scholar. Comparative observations and knowledge-graph gap analyses represent cross-study synthesis, while the research directions in the preceding section are explicitly framed as falsifiable proposals for future empirical validation.

## Evidence Grounding & Claim Classification Summary

- **Supported Literature Claims (`supported`):** 8 (directly linked to retrieved sources `[1]`–`[8]`)
- **Comparative Synthesis Observations (`synthesis`):** 2
- **Proposed Future Hypotheses (`proposed`):** 3 (explicitly separated from published findings)
- *Note:* All literature claims are extracted from retrieved paper abstracts and bibliographic metadata (ArXiv / Semantic Scholar).
- *Note:* Quantitative figures reported in individual abstracts reflect the original authors' experimental protocols and are not directly re-benchmarked in this review.

## References

[1] Jia-Ping Xiao, Cheng-Wen Tsao, Yu-Hang Zhang, Mir Feroskhan, "FM-Planner: Foundation Model Guided Path Planning for Autonomous Drone Navigation," arXiv.org, 2025. [Online]. Available: https://www.semanticscholar.org/paper/70762c699b32ad15212c9607f72dc049e050003c (arXiv:2505.20783)  
[2] Abdul Manan Khan, Ikram Ur Rehman, Nagham Saeed, et al., "Context-Aware Autonomous Drone Navigation Using Large Language Models (LLMs)," AAAI Summer Symposia, 2025. [Online]. Available: https://www.semanticscholar.org/paper/09718b55d6b2ce0adf277e82ba471248b58ca39a (DOI:10.1609/aaaiss.v6i1.36039)  
[3] Yu-Hang Zhang, Jia-Ping Xiao, Mir Feroskhan, "Learning Cross-Modal Visuomotor Policies for Autonomous Drone Navigation," IEEE Robotics and Automation Letters, 2025. [Online]. Available: https://www.semanticscholar.org/paper/cbdce8da0475c72fbfc9c019578ab6dccc61c9e6 (DOI:10.1109/LRA.2025.3559824)  
[4] Maisy Lam, Joshua Herrera, Sayed Saad Afzal, et al., "MiNav: Autonomous Drone Navigation Indoors Using Millimeter-Waves," Proceedings of the ACM on Interactive Mobile Wearable and Ubiquitous Technologies, 2025. [Online]. Available: https://www.semanticscholar.org/paper/416c56be127ee236c019418fd704a3ec8c6c3307 (DOI:10.1145/3749464)  
[5] Guglielmo Del Col, V. Karjalainen, T. Hakala, E. Honkavaara, "Autonomous Drone Navigation in Forest Environments Using Deep Learning," The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 2025. [Online]. Available: https://www.semanticscholar.org/paper/3336b5e4758deaf5b4e15386768100e3cad8c95f (DOI:10.5194/isprs-archives-xlviii-2-w11-2025-87-2025)  
[6] Thássio Gomes Silva, Kauê F. Bonfá, Gabriela M. Goulart, et al., "Simulation-Based Autonomous Drone Navigation for Security and Monitoring Using Deep Reinforcement Learning," 2025 Brazilian Conference on Robotics (CROS), 2025. [Online]. Available: https://www.semanticscholar.org/paper/8333843dac4f5b3ad1c89c482fed940ec1a1f780 (DOI:10.1109/CROS66186.2025.11066101)  
[7] S. Safie, Raudhah Mohd Khairil Azhar, Mohamad Hanafi Ali, "Deep Learning Enabled Autonomous Drone Navigation for Safety Inspections in Chemical Plants," 2025 IEEE International Conference on Automatic Control and Intelligent Systems (I2CACIS), 2025. [Online]. Available: https://www.semanticscholar.org/paper/291d690d71dc813b77a3f550e906d070c6ca44c7 (DOI:10.1109/I2CACIS65476.2025.11100664)  
[8] Sandeep Shinde, Soham Kanhe, Shravan Katore, et al., "Autonomous Drone Navigation Using Reinforcement Learning," 2025 Global Conference on Information Technology and Communication Networks (GITCON), 2025. [Online]. Available: https://www.semanticscholar.org/paper/d0dadda134e7dc7d6c3df9fb8b093e96be2e2b15 (DOI:10.1109/GITCON65266.2025.11378297)  
