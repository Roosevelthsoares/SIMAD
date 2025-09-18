# -*- coding: utf-8 -*-
"""
Experimento de Simulação – Tempo até a Venda (TOM)
- Chegadas de interessados: processo de Poisson (interarrivos Exponenciais)
- Negociação: Cadeia de Markov em tempo discreto com estados absorventes (Sold, WalkAway)

Saídas:
  - Resumo impresso no console (média, desvio, IC95%, comparação com teoria)
  - CSV por cenário com as amostras do TOM (./resultados/<cenario>.csv)
  - (Opcional) gráfico de CDF empírica (requer matplotlib)

Pontos de customização marcados com "👉 MUDE AQUI".
"""

import os
import numpy as np
import pandas as pd

# ==============================
# 1) PARÂMETROS GERAIS  👉 MUDE AQUI
# ==============================
SEED = 123
N_REPLICAS = 10000          # nº de simulações por cenário  👉 MUDE AQUI se quiser mais/menos precisão
GERAR_GRAFICO = True        # requer matplotlib; coloque False se não quiser gráficos

# ==============================
# 2) CENÁRIOS (λ e τ)  👉 MUDE AQUI
# ==============================
# λ = taxa de chegadas (por dia) | τ = duração média de cada rodada de barganha (em dias)
CENARIOS = {
    "demanda_alta":  {"lambda": 0.08, "tau": 1.0},
    "demanda_baixa": {"lambda": 0.04, "tau": 1.0},
    # Adicione aqui outros cenários, por exemplo:
    # "mercado_moderado": {"lambda": 0.06, "tau": 0.8},
}

# ==============================
# 3) MATRIZ DE TRANSIÇÃO P  👉 MUDE AQUI
# ==============================
# Ordem dos estados: [S0, S1, S2, Sold, WalkAway]
# - Ajuste os números abaixo para refletir sua política de negociação.
# - Cada linha precisa somar 1. Os estados "Sold" e "WalkAway" são absorventes (linha identidade).
P = np.array([
    [0.55, 0.25, 0.00, 0.12, 0.08],  # S0: fica, vai a -5%, vende, desiste
    [0.20, 0.40, 0.25, 0.10, 0.05],  # S1: volta S0, fica, vai a -10%, vende, desiste
    [0.00, 0.15, 0.35, 0.35, 0.15],  # S2: volta S1, fica, vende, desiste
    [0.00, 0.00, 0.00, 1.00, 0.00],  # Sold (absorvente)
    [0.00, 0.00, 0.00, 0.00, 1.00],  # WalkAway (absorvente)
], dtype=float)

# Índices dos estados (se mudar a ordem em P, atualize aqui)
S0, S1, S2, SOLD, WALK = 0, 1, 2, 3, 4
TRANSIENTES = [S0, S1, S2]
ABSORVENTES = [SOLD, WALK]
ESTADO_INICIAL = S0

# ==============================
# 4) FUNÇÕES DE APOIO (teoria)
# ==============================

def matriz_fundamental(P, transientes, absorventes):
    """Retorna T, R, N, B, E_steps para a cadeia de Markov."""
    T = P[np.ix_(transientes, transientes)]
    R = P[np.ix_(transientes, absorventes)]
    I = np.eye(T.shape[0])
    N = np.linalg.inv(I - T)         # matriz fundamental
    B = N @ R                        # prob. de absorção
    ones = np.ones((T.shape[0], 1))
    E_steps = N @ ones               # passos esperados até absorção
    return T, R, N, B, E_steps

def teoria_markov(P, transientes, absorventes, estado_inicial, lmbda, tau):
    """Calcula π_sold, E[S], E[Ciclo] e E[TOM] teórico para dados λ e τ."""
    _, _, N, B, E_steps = matriz_fundamental(P, transientes, absorventes)
    pi_sold = float(B[transientes.index(estado_inicial), 0])   # coluna 0 = Sold
    E_S = float(E_steps[transientes.index(estado_inicial), 0]) # passos esperados desde S0
    E_ciclo = 1.0 / lmbda + tau * E_S
    E_TOM = (1.0 / pi_sold) * E_ciclo
    return pi_sold, E_S, E_ciclo, E_TOM

# ==============================
# 5) SIMULAÇÃO (uma réplica)
# ==============================

def simula_uma_replicacao_TOM(lmbda, tau, P, estado_inicial, sold_idx, walk_idx, rng):
    """Simula o TOM completo: espera por chegada + negociação Markov até vender.
       Se a negociação terminar em WalkAway, inicia novo ciclo e segue até vender.
    """
    n_states = P.shape[0]
    tempo = 0.0

    while True:
        # Chegada do próximo interessado (interarrival ~ Exponencial(λ))
        interarrival = rng.exponential(scale=1.0 / lmbda)
        tempo += interarrival

        # Negociação Markov
        estado = estado_inicial
        while True:
            # 1 rodada de barganha consome τ
            tempo += tau

            # Decide próximo estado conforme a linha de P
            probs = P[estado]
            estado = rng.choice(n_states, p=probs)

            if estado == sold_idx:
                return tempo          # vendeu: TOM finalizado
            if estado == walk_idx:
                break                 # desistiu: volta para esperar nova chegada

# ==============================
# 6) EXECUÇÃO DOS CENÁRIOS
# ==============================

def ic95(vetor):
    n = len(vetor)
    m = float(np.mean(vetor))
    sd = float(np.std(vetor, ddof=1))
    half = 1.96 * sd / np.sqrt(n)
    return m, sd, (m - half, m + half)

def executa_cenario(nome, params, rng):
    lmbda = params["lambda"]
    tau = params["tau"]

    # Teoria
    pi_sold, E_S, E_ciclo, E_TOM_teo = teoria_markov(
        P, TRANSIENTES, ABSORVENTES, ESTADO_INICIAL, lmbda, tau
    )

    # Simulação
    amostras = np.array([
        simula_uma_replicacao_TOM(lmbda, tau, P, ESTADO_INICIAL, SOLD, WALK, rng)
        for _ in range(N_REPLICAS)
    ])
    media, dp, (lo, hi) = ic95(amostras)
    erro_rel = abs(media - E_TOM_teo) / E_TOM_teo * 100.0

    # Salvar CSV
    os.makedirs("resultados", exist_ok=True)
    csv_path = os.path.join("resultados", f"{nome}.csv")
    pd.DataFrame({"TOM_dias": amostras}).to_csv(csv_path, index=False)

    # Gráfico (opcional)
    if GERAR_GRAFICO:
        try:
            import matplotlib.pyplot as plt
            xs = np.sort(amostras)
            F = np.arange(1, len(xs) + 1) / len(xs)
            plt.figure(figsize=(6,4))
            plt.step(xs, F, where="post", label="CDF empírica")
            plt.xlabel("Tempo até a venda (dias)")
            plt.ylabel("F(t)")
            plt.title(f"CDF – {nome}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plot_path = os.path.join("resultados", f"cdf_{nome}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=160)
            plt.close()
        except Exception as e:
            print("Aviso: não foi possível gerar o gráfico (matplotlib).", e)

    # Resumo
    resumo = {
        "cenario": nome,
        "lambda": lmbda,
        "tau": tau,
        "pi_sold_teorico": pi_sold,
        "E[S]_teorico": E_S,
        "E[Ciclo]_teorico": E_ciclo,
        "E[TOM]_teorico": E_TOM_teo,
        "E[TOM]_simulado": media,
        "DP": dp,
        "IC95_inf": lo,
        "IC95_sup": hi,
        "erro_rel_%": erro_rel,
        "csv": csv_path
    }
    return resumo

def comparar_cenarios():
    import json
    import matplotlib.pyplot as plt

    # Carrega o resumo
    df = pd.read_csv("resultados/resumo_resultados.csv")

    # Garante que existem os cenários esperados
    req = {"demanda_alta", "demanda_baixa"}
    if not req.issubset(set(df["cenario"])):
        print("Aviso: cenários 'demanda_alta' e 'demanda_baixa' não encontrados no resumo.")
        return

    # Junta lado a lado (usando sufixos _alta/_baixa)
    da = df[df["cenario"]=="demanda_alta"].iloc[0]
    db = df[df["cenario"]=="demanda_baixa"].iloc[0]

    comp = {
        "lambda_alta": da["lambda"], "lambda_baixa": db["lambda"],
        "tau_alta": da["tau"], "tau_baixa": db["tau"],
        "pi_sold": da["pi_sold_teorico"],               # igual nos dois cenários
        "E[S]": da["E[S]_teorico"],                      # igual nos dois cenários
        "E[Ciclo]_alta": da["E[Ciclo]_teorico"],
        "E[Ciclo]_baixa": db["E[Ciclo]_teorico"],
        "E[TOM]_teorico_alta": da["E[TOM]_teorico"],
        "E[TOM]_teorico_baixa": db["E[TOM]_teorico"],
        "E[TOM]_sim_alta": da["E[TOM]_simulado"],
        "E[TOM]_sim_baixa": db["E[TOM]_simulado"],
    }
    comp["var_%_E[Ciclo]"] = (comp["E[Ciclo]_baixa"] - comp["E[Ciclo]_alta"]) / comp["E[Ciclo]_alta"] * 100.0
    comp["var_%_E[TOM]_teorico"] = (comp["E[TOM]_teorico_baixa"] - comp["E[TOM]_teorico_alta"]) / comp["E[TOM]_teorico_alta"] * 100.0
    comp["var_%_E[TOM]_sim"] = (comp["E[TOM]_sim_baixa"] - comp["E[TOM]_sim_alta"]) / comp["E[TOM]_sim_alta"] * 100.0

    comp_df = pd.DataFrame([comp])
    comp_df.to_csv("resultados/comparacao_alta_vs_baixa.csv", index=False)
    print("\n=== COMPARAÇÃO ALTA vs BAIXA ===")
    print(comp_df.T.to_string(header=False))

    # CDF dupla no mesmo gráfico
    alta = pd.read_csv("resultados/demanda_alta.csv")["TOM_dias"].values
    baixa = pd.read_csv("resultados/demanda_baixa.csv")["TOM_dias"].values
    for data, label in [(np.sort(alta), "Demanda alta"), (np.sort(baixa), "Demanda baixa")]:
        F = np.arange(1, len(data)+1) / len(data)
        plt.step(data, F, where="post", label=label)
    plt.xlabel("Tempo até a venda (dias)")
    plt.ylabel("F(t)")
    plt.title("CDF – Alta vs Baixa")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultados/cdf_comparacao_alta_baixa.png", dpi=160)
    plt.close()

# Chame no final do main():
def main():
    rng = np.random.default_rng(SEED)
    resumos = []
    for nome, params in CENARIOS.items():
        resumos.append(executa_cenario(nome, params, rng))
    df = pd.DataFrame(resumos)
    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== RESUMO DOS RESULTADOS ===")
    print(df.to_string(index=False))
    os.makedirs("resultados", exist_ok=True)
    df.to_csv("resultados/resumo_resultados.csv", index=False)
    comparar_cenarios()   # <<< adiciona esta linha


# def main():
#     rng = np.random.default_rng(SEED)
#     resumos = []
#     for nome, params in CENARIOS.items():
#         resumos.append(executa_cenario(nome, params, rng))

#     df = pd.DataFrame(resumos)
#     pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
#     print("\n=== RESUMO DOS RESULTADOS ===")
#     print(df.to_string(index=False))

#     os.makedirs("resultados", exist_ok=True)
#     df.to_csv("resultados/resumo_resultados.csv", index=False)
#     print("\nArquivos salvos em ./resultados/")

if __name__ == "__main__":
    main()
