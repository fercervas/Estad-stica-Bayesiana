# Examen práctico 
# Cervantes Vasconcelos María Fernanda

using CSV, DataFrames, Plots, LaTeXStrings, Distributions, QuadGK
ma = CSV.read("318711827.csv", DataFrame)
include("Exponencial.jl")


# a) Calcular estimaciones puntuales a posteriori (no informativas) para cada parámetro, 
# y para una observación futura (predicción), utilizando tanto la mediana como la media en cada caso.
X = bExpo(ma[:,1])
keys(X)

# Estimación puntual para μ (Mediana)
# `qμ` = función de cuantiles a posteriori marginal para μ
μM = X.qμ(0.5)
println("La estimación puntual de µ vía la mediana es ", μM)

# Estimación puntual para μ (Media)
# `dμ` = función de densidad a posteriori marginal para μ
μE, err = quadgk(x -> x .* X.dμ.(x), -Inf, Inf, rtol=1e-8)
println("La estimación puntual de µ vía la media es ", μE)

# Estimación puntual para θ (Mediana)
# `qθ` = función de cuantiles a posteriori marginal para θ
θM = X.qθ(0.5)
println("La estimación puntual de θ vía la mediana es ", θM)

# Estimación puntual para θ (Media)
# `dθ` = función de densidad a posteriori marginal para θ

θE, err = quadgk(x -> x .* X.dθ.(x), 0, Inf, rtol=1e-8)
println("La estimación puntual de θ vía la media es ", μE)

# observación futura
# `dp` = función de densidad predictiva a posteriori
# `qp` = función de cuantiles predictiva a posteriori  

XM = X.qp(0.5)
println("La estimación puntual de X_{n+1} vía la mediana es ", XM)
XE, err = quadgk(x -> x .* X.dp.(x), -Inf, Inf, rtol=1e-8)
println("La estimación puntual de X_{n+1} vía la media es ", XE)


# b) Graficar la densidad a posteriori (no informativa) del parámetro de localización, 
# y agregar líneas verticales con los valores tanto de la mediana como de la media 
# (identificados claramente) e indicando ahí mismo sus valores.
# El parámetro de localización es µ


begin
    t = range(X.qμ(0.001), X.qμ(0.999), length = 1_000)
    plot(t, X.dμ.(t), lw = 3, label = "a posteriori no info", color = :red, title="Densidad a posteriori (no informativa) del parámetro μ", titlefontsize=12)
    xaxis!(L"μ")
    yaxis!("densidad")
    μM1 = round(μM, digits=4)
    μE1 = round(μE, digits=4)
    scatter!([μM], [0], ms = 5, color = :blue, label = "μ_Mediana = $μM1")
    scatter!([μE], [0], ms = 5, color = :green, label = "μ_Media = $μE1")
    # Líneas verticales
    vline!([μM], linestyle=:dash, lw=2, color=:blue, label=false)
    vline!([μE], linestyle=:dash, lw=2, color=:green, label=false)
end


# c) Graficar la densidad a posteriori (no informativa) del parámetro de precisión, 
# y agregar líneas verticales con los valores tanto de la mediana como de la media 
# (identificados claramente) e indicando ahí mismo sus valores.
# El parámetro de precisión es θ


begin
    t = range(X.qθ(0.001), X.qθ(0.999), length = 1_000)
    plot(t, X.dθ.(t), lw = 3, label = "a posteriori no info", color = :red, title="Densidad a posteriori (no informativa) del parámetro θ", titlefontsize=12)
    xaxis!(L"θ")
    yaxis!("densidad")
    θM1 = round(θM, digits=4)
    θE1 = round(θE, digits=4)
    scatter!([θM], [0], ms = 5, color = :blue, label = "θ_Mediana = $θM1")
    scatter!([θE], [0], ms = 5, color = :green, label = "θ_Media = $θE1")
    # Líneas verticales
    vline!([θM], linestyle=:dash, lw=2, color=:blue, label=false)
    vline!([θE], linestyle=:dash, lw=2, color=:green, label=false)
end


# d) Graficar la densidad predictiva a posteriori (no informativa), 
# y agregar líneas verticales con los valores tanto de la mediana como de la media 
# (identificados claramente) e indicando ahí mismo sus valores.

begin
    x = collect(range(X.qp(0.001), X.qp(0.99), length = 1_000))
    plot(x, X.dp.(x), lw = 1.7, color = :red, label = "predictiva", title="Densidad predictiva a posteriori (no informativa)", titlefontsize=12)
    xaxis!(L"x")
    yaxis!("densidad")
    XM1 = round(XM, digits=4)
    XE1 = round(XE, digits=4)
    scatter!([XM], [0], ms = 5, color = :blue, label = "X_Mediana = $XM1")
    scatter!([XE], [0], ms = 5, color = :green, label = "X_Media = $XE1")
    # Líneas verticales
    vline!([XM], linestyle=:dash, lw=2, color=:blue, label=false)
    vline!([XE], linestyle=:dash, lw=2, color=:green, label=false)
end


# e) Graficar con colores los conjuntos de nivel de la densidad conjunta a posteriori de los parámetros, 
# y agregar un punto que se distinga claramente con coordenadas igual a la  mediana a posteriori de cada parámetro.

begin
    sim_μθ = X.r(10_000)
    ngrid = 100
    x = collect(range(quantile(sim_μθ[:, 1], 0.03), maximum(sim_μθ[:, 1]), length = ngrid))
    y = collect(range(minimum(sim_μθ[:, 2]), maximum(sim_μθ[:, 2]), length = ngrid))
    z = zeros(ngrid, ngrid)
    for i ∈ eachindex(x), j ∈ eachindex(y)
        z[j, i] = X.d(x[i], y[j])
    end
    contour(x, y, z, xlabel = L"μ", ylabel = L"θ", fill = true, 
            title = "Conjuntos de nivel de la densidad conjunta a posteriori", size = (500,500), color=:viridis, titlefontsize=11 )
    scatter!([μM], [θM], ms = 5, mc = :red, label = "Mediana-(μ,θ) = ($μM1,$θM1)")    
    scatter!([μE], [θE], ms = 5, mc = :orange, label = "Media-(μ,θ) = ($μE1,$θE1)")    
end


# f) Simular a partir de la distribución conjunta a posteriori de los parámetros una muestra aleatoria bivariada de tamaño 
# 3 mil, y graficar los puntos obtenidos en un diagrama de dispersión (scatter plot), y agregar un punto con color distinto 
# con coordenadas igual a la mediana a posteriori de cada parámetro.

# Simulación de tamaño 3_000
sim_μθ = X.r(3_000)
begin
    scatter(sim_μθ[:, 1], sim_μθ[:, 2], ms = 1.5, mc = :green, size = (500,500), label = "")
    title!("Simulación conjunta a posteriori")
    xaxis!(L"μ")
    yaxis!(L"θ")
    scatter!([μM], [θM], ms = 5, mc = :red, label = "Mediana-(μ,θ) = ($μM1,$θM1)")
    scatter!([μE], [θE], ms = 5, mc = :yellow, label = "Media-(μ,θ) = ($μE1,$θE1)")
end


