for i in `seq 0 4`; do
    echo $1
    python run_gym_rllib_agent_blacklist.py configs/config1 2 1 $i &
done
