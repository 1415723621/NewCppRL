import gymnasium.envs.registration

gymnasium.envs.registration.register(
    id="Pasture-v1",
    entry_point="envs.cpp_env_v1:CppEnv",
)
gymnasium.envs.registration.register(
    id="Pasture-v2",
    entry_point="envs.cpp_env_v2:CppEnv",
)
gymnasium.envs.registration.register(
    id="Pasture-v3",
    entry_point="envs.cpp_env_v3:CppEnv",
)
