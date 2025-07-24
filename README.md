# DRL for DDFJSP
# # Introduction
  Dear readers, thank you for your esteemed visit. This repository will share the full code of the paper ''Dynamic Scheduling for Distributed Flexible Job Shop Using Deep Reinforcement Learning'' after it is accepted by the reviewing journal. In the following section, we describe the content of these code files in detail.
# # # Description of code files
# # # # Code files in the folder ''Drl''
1. Distributed_flexible_job_shop.py. It is used for establishing an distributed flexible job shop scheduling environment. This environment contains the updating function of state features, the action function, and the scheduling function.
2. Objection_for_DDFJSP.py. It is used for creating job and machine objects and contains their data updating functions.
3. Job_creator.py. It is responsible for generating instances under different production configurations.
4. D3QN.py. It contains an D3QN class with an experience replay buffer and a target network. Grounded in this, a D3QN-based dynamic scheduling framework is established. At the end of this file, the ''argparse'' is employed to convenient the parameter setting.
5. Composite_rule.py. It encompasses an evaluate function for all composite rules, which would calculate and save the results of composite rules.
6. Distributed_flexible_job_shop_P.py. It has a structure similar to the "Distributed_flexible_job_shop.py", but contains all combined priority dispatching rules (PDRs).
7. Priority_dispatching_rule.py. It owns the similar structure and function with ''Composite_rule.py'' and is utilized for evaluating combined PDRs.
8. As for other code files namely DRL_Chang.py, DRL_Gui.py, DRL_Luo.py, DRL_Wang.py, DRL_Zhang.py, and DRL_Lei, they are the comparative algorithms in Section 6.3.3.
# # # # Code files in the folder ''Metaheuristics''
1. Distributed_flexible_job_shop_L. It is used for establishing an distributed flexible job shop scheduling environment. This environment contains the information updating function. The structure and function of Distributed_flexible_job_shop_Z.py is similar to it.
2. Encode_L.py and Encode_Z.py. They contain the ''encode'' class and are used for generating the initial population.
3. Decode_L.py and Decode_Z.py. They include the ''decode'' class and are responsible for decoding the chromosomes of the population.
4. Job_L.py and Job_Z.py. They are used for creating the job objects and contain the functions for inputting and deleting information.
5. Machine_L.py and Machine_Z.py. They are used for creating machine objects and contain the functions for inputting and deleting information.
6. Job_creator.py. It is responsible for generating instances under different production configurations.
7. FastNDSort.py. It provides non-dominated sorting for solution sets.
8. MA_L.py and MA_Z.py. They contain the memetic algorithm class, fitness calculation functions, crossover functions, mutation functions, and local search functions.
9. MA_L.py. It is the main algorithm for rescheduling.
10. MA_Z.py. It is the main algorithm for rescheduling.
# # # # Files in the folder ''Cplex''
1. 222.dat. It is the prameters of intance 222.
2. 222.mod. It is the CPLEX codes of intance 222.
3. 233.dat. It is the prameters of intance 233.
4. 233.mod. It is the CPLEX codes of intance 233.
5. 322.dat. It is the prameters of intance 322.
6. 322.mod. It is the CPLEX codes of intance 322.
7. 333.dat. It is the prameters of intance 333.
8. 333.mod. It is the CPLEX codes of intance 333.
9. 422.dat. It is the prameters of intance 422.
10. 422.mod. It is the CPLEX codes of intance 422.
11. 433.dat. It is the prameters of intance 433.
12. 433.mod. It is the CPLEX codes of intance 433.
