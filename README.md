# Trombone-Slide-Position-Optimizer
I built this app to figure out the most physically efficient slide paths for complex trombone licks. It treats playing the horn like a logistics routing problem and uses linear programming to find the best balance between slide speed, valve usage, and tone quality.
Key Features
Multi-Objective Optimization Engine: Uses the PuLP linear programming library to minimize a custom cost function that weighs total slide travel (arm movement) against valve actuations (thumb movement) and awkward partial leaps (lip fatigue).
Interactive UI: A Streamlit web dashboard that allows users to dynamically adjust optimization penalty weights and visualize the resulting physical path in real-time.
Relational Data Architecture: Uses Pandas to manage a scalable data warehouse mapping different instrument configurations (Straight Tenor vs. Symphony Tenor F-Attachment) to their respective acoustic properties and basic position defaults.
Custom Syntax Parser: Allows users to input standard scientific pitch notation, while supporting a custom syntax to pass Hard Constraints to the math solver (e.g., forcing specific alternate positions).
Efficiency Analytics: Automatically calculates and displays metrics comparing the optimized route against a standard "basic" route, showing percentage reductions in travel distance and mechanical complexity.
How the Math Works
The engine uses a MILP solver to balance conflicting priorities using Soft Constraints:
Distance Penalty: Minimizes the raw physical distance the slide must travel.
Direction Change Penalty: Penalizes the "wasted energy" of stopping and reversing the slide's momentum.
Valve/Trigger Penalty: Assigns a cost to engaging the F-attachment, which requires thumb coordination.
Alternate Position Penalty: A user-adjustable "resonance penalty" that discourages the use of stuffy alternate positions unless they offer a massive mechanical advantage over the basic open-horn position.
Usage & Syntax
Users can enter standard MIDI sequences (e.g., Bb2 F3 Bb3 D4).
The parser also accepts Hard Constraints using asterisks to force the solver's behavior on specific notes:
*C3 → Forces the algorithm to use the standard Basic Position for this note.
13*F3 → Forces a specific coordinate: Valve 1 (Trigger engaged) and Visual Position 3.
The solver will lock in these constraints and route the rest of the sequence around them as efficiently as possible.
