## Caso: μ ∈ ℜ desconocida, θ > 0 desconocida
"""
    bExpo(xobs)

    Modelo bayesiano con distribuciones a posteriori y predictiva a posteriori no informativas
    para una distribución Exponencial(μ,θ), donde:

    ```math 
    f(x|μ,θ) = θexp(-θ(x - μ)),  x > μ
    ```

    donde:

    - `xobs` = vector con la muestra aleatoria observada

    Entrega un tupla etiquetada con los siguientes elementos:

    1. `familia` = distribución de probabilidad
    2. `dμ` = función de densidad a posteriori marginal para μ
    3. `pμ` = función de distribución a posteriori marginal para μ
    4. `qμ` = función de cuantiles a posteriori marginal para μ
    5. `rμ` = función simuladora a posteriori marginal para μ
    6. `dθ` = función de densidad a posteriori marginal para θ
    7. `pθ` = función de distribución a posteriori marginal para θ
    8. `qθ` = función de cuantiles a posteriori marginal para θ
    9. `rθ` = función simuladora a posteriori marginal para θ
    10. `dμ_θ` = función de densidad condicional a posteriori para μ dado θ
    11. `pμ_θ` = función de distribución condicional a posteriori para μ dado θ
    12. `qμ_θ` = función de cuantiles condicional a posteriori para μ dado θ
    13. `rμ_θ` = función simuladora condicional a posteriori para μ dado θ
    14. `dθ_μ` = función de densidad condicional a posteriori para θ dado μ
    15. `pθ_μ` = función de distribución condicional a posteriori para θ dado μ
    16. `qθ_μ` = función de cuantiles condicional a posteriori para θ dado μ
    17. `rθ_μ` = función simuladora condicional a posteriori para θ dado μ
    18. `d` = función de de densidad conjunta a posteriori para (μ,θ)
    19. `r` = función simuladora a posteriori para (μ,θ) 
    20. `dp` = función de densidad predictiva a posteriori
    21. `pp` = función de distribución predictiva a posteriori
    22. `qp` = función de cuantiles predictiva a posteriori 
    23. `rp` = función simuladora predictiva a posteriori 
    24. `n` = tamaño de la muestra observada 
    25. `muestra` = vector de la muestra observada
    26. `sx` = suma muestral 
    27. `xmin` = mínimo muestral

    # Ejemplo
    ```
    μ,θ = -1.5, 3.7; # valor teórico de los parámetros desconocidos
    X = vaExponencial(μ,θ);
    n = 1_000; # tamaño de muestra a simular
    xx = X.sim(n) # simular muestra observada
    ## modelo bayesiano:
    post = bExpo(xx);
    keys(post)
    post.familia
    ## estimación puntual marginal de μ vía la mediana:
    μ # valor teórico
    post.qμ(0.5) # a posteriori no informativa
    ## estimación puntual marginal de θ vía la mediana:
    θ # valor teórico
    post.qθ(0.5) # a posteriori no informativa
    ## simulaciones de (μ,θ) y algunas verificaciones marginales
    sim_μθ = post.r(10_000); # simulación de conjunta (μ,θ)
    sim_μ = post.rμ(10_000); # simulación marginal de μ
    sim_θ = post.rθ(10_000); # simulación marginal de θ
    μ # valor teórico
    median(sim_μθ[:, 1]) # estimación puntual de μ vía simulación
    median(sim_μ) # estimación puntual de μ vía simulación
    θ # valor teórico 
    median(sim_μθ[:, 2]) # estimación puntual de θ vía simulación
    median(sim_θ) # estimación puntual de θ vía simulación
    ## estimación puntual predictiva de la mediana:
    X.mediana # teórica
    post.qp(0.5) # estimación
    ```
"""
function bExpo(xobs)
    n = length(xobs)
    sx = sum(xobs)
    xmin = minimum(xobs)
    # modelo π
    πd(μ, α, β) = β*exp(β*(μ - α))*(μ < α)
    πp(t, α, β) = (t < α)*exp(-β*(α - t)) + 1*(t ≥ α)
    πq(u, α, β) = (α + log(u)/β)*(0 ≤ u ≤ 1)
    πr(m, α, β) = πq.(rand(m), α, β)
    # a posteriori marginal para μ
    dpostμ(t) = (((sx - n*xmin)/(sx - n*t))^n)*((n^2)/(sx - n*t))*(t < xmin)
    ppostμ(t) = (((sx - n*xmin)/(sx - n*t))^n)*(t < xmin) + 1*(t ≥ xmin)
    function qpostμ(u)
        if 0 ≤ u ≤ 1
            return (sx - (sx - n*xmin)/(u^(1/n))) / n
        else
            return NaN 
        end
    end
    rpostμ(m) = qpostμ.(rand(m)) 
    # a posteriori marginal para θ
    G = Gamma(n, 1 / (sx - n*xmin)) 
    dpostθ(t) = pdf(G, t)
    ppostθ(t) = cdf(G, t)
    qpostθ(u) = quantile(G, u)
    rpostθ(m) = rand(G, m)
    # a posteriori condicional de μ dado θ
    dpostμ_θ(t, θ) = πd(t, xmin, n*θ)
    ppostμ_θ(t, θ) = πp(t, xmin, n*θ)
    qpostμ_θ(u, θ) = πq(u, xmin, n*θ)
    rpostμ_θ(m, θ) = πr(m, xmin, n*θ)
    # a posteriori condicional de θ dado μ
    dpostθ_μ(t, μ) = pdf(Gamma(n+1, 1/(sx - n*μ)), t)
    ppostθ_μ(t, μ) = cdf(Gamma(n+1, 1/(sx - n*μ)), t)
    qpostθ_μ(u, μ) = quantile(Gamma(n+1, 1/(sx - n*μ)), u)
    rpostθ_μ(m, μ) = rand(Gamma(n+1, 1/(sx - n*μ)), m)
    # a posteriori conjunta de (μ,θ)
    dpost(μ,θ) = pdf(G, θ) * πd(μ, xmin, n*θ)
    function rpost(m)
        θ = rpostθ(m)
        μ = zeros(m)
        for i ∈ 1:m
            μ[i] = rpostμ_θ(1, θ[i])[1]
        end
        return [μ θ]
    end
    # predictiva a posteriori
    function dpred(x)
        if x ≤ xmin 
            return (n/(n+1)) * ((sx - n*xmin)/(sx - n*x))^n * (n/(sx - n*x))
        else
            return (n/(n+1)) * ((sx - n*xmin)/(x + sx - (n+1)*xmin))^n * (n/(x + sx - (n+1)*xmin))
        end
    end
    function ppred(t)
        if t ≤ xmin 
            return (1/(n+1)) * ((sx - n*xmin)/(sx - n*t))^n
        else
            return 1 - (n/(n+1))*((sx - n*xmin)/(t + sx - (n+1)*xmin))^n
        end
    end
    function qpred(u)
        if 0 ≤ u ≤ 1/(n+1)
            return ( sx - (sx - n*xmin)/(((n+1)*u)^(1/n)) ) / n
        elseif 1/(n+1) < u ≤ 1 
            return (sx - n*xmin)/(((1 + 1/n)*(1 - u))^(1/n)) - sx + (n+1)*xmin 
        else
            return NaN 
        end
    end
    rpred(m) = qpred.(rand(m))
    return (familia = "Exponencial(μ,θ)", dμ = dpostμ, pμ = ppostμ, qμ = qpostμ, 
            rμ = rpostμ, dθ = dpostθ, pθ = ppostθ, qθ = qpostθ, rθ = rpostθ, 
            dμ_θ = dpostμ_θ, pμ_θ = ppostμ_θ, qμ_θ = qpostμ_θ, rμ_θ = rpostμ_θ,
            dθ_μ = dpostθ_μ, pθ_μ = ppostθ_μ, qθ_μ = qpostθ_μ, rθ_μ = rpostθ_μ,
            d = dpost, r = rpost, dp = dpred, pp = ppred, qp = qpred, rp = rpred,
            n = n, muestra = xobs, sx = sx, xmin = xmin)
end






