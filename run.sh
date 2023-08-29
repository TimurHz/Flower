#!/bin/bash

echo -e "\033[35m starting simulation \033[0m"

echo -e "\033[32m client:10    batch:32    runden:5    model:1 \033[0m"
#client:10    batch:32    runden:5    model:1    
python3 main.py batch_size=32 num_rounds=5

echo -e "\033[32m client:10    batch:32    runden:10    model:1 \033[0m"
#client:10    batch:32    runden:10    model:1    
python3 main.py batch_size=32 num_rounds=10

echo -e "\033[32m client:10    batch:32    runden:15    model:1 \033[0m"
#client:10    batch:32    runden:15    model:1    
python3 main.py batch_size=32 num_rounds=15

#-----------

echo -e "\033[32m client:10    batch:64    runden:5    model:1 \033[0m"
#client:10    batch:64    runden:5    model:1    
python3 main.py batch_size=64 num_rounds=5

echo -e "\033[32m client:10    batch:64    runden:10    model:1 \033[0m"
#client:10    batch:64    runden:10    model:1    
python3 main.py batch_size=64 num_rounds=10

echo -e "\033[32m client:10    batch:64    runden:15    model:1 \033[0m"
#client:10    batch:64    runden:15    model:1    
python3 main.py batch_size=64 num_rounds=15

