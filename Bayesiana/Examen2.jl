# Cervantes Vasconcelos María Fernanda
# Examen práctico estadística bayesiana


# a) Calcular estimaciones puntuales a posteriori para cada parámetro, 
# y para una observación futura (predicción), utilizando la media en cada caso.

# b) Graficar las densidades a posteriori de cada parámetro, y en cada caso agregar 
# una línea vertical con el valor de la media a posteriori, e indicando ahí mismo su valor.

# c) Graficar la densidad predictiva a posteriori, y agregar una línea vertical con el valor
# de la media predictiva a posteriori e indicando ahí mismo su valor.

using CSV, DataFrames, Plots, LaTeXStrings, Distributions, QuadGK
ma = CSV.read("ma.csv", DataFrame)

xobs = ma[:,1]
mean(xobs), var(xobs) # estimaciones muestrales de E(X) y V(X)
function momαβ(μ,σ²) 
    # valor de los párametros dados E(X) y V(X)
    α = μ * ((μ - μ^2)/σ² - 1)
    β = α/μ - α
    return (α, β)
end

## Método ABC 

function ABC(muestra::Vector, nsim = 1_000_000, nselec = 100)
    nobs = length(muestra)
    δ(u, v) = √sum((u - v) .^2) # distancia euclidiana
    dist = zeros(nsim) # inicializar vector de distancias
    αmom, βmom = momαβ(mean(muestra), var(muestra))
    priorα = rand(Uniform(0, max(1.0, 2*αmom)), nsim)
    priorβ = rand(Uniform(0, max(1.0, 2*βmom)), nsim)

    for i ∈ 1:nsim 
        xsim = rand(Beta(priorα[i], priorβ[i]), nobs)
        αmomsim, βmomsim = momαβ(mean(xsim), var(xsim))
        dist[i] = δ([αmomsim, βmomsim], [αmom, βmom])
    end
    iSelec = sortperm(dist)[1:nselec] # simulaciones a seleccionar
    postα = priorα[iSelec]
    postβ = priorβ[iSelec]
    
    return (α = postα, β = postβ)
end

@time post = ABC(xobs, 1_000_000, 10_000);  
# Estimaciones de los parámetros a posteriori de α y β
βpost = mean(post.β) # media a posteriori de β
αpost = mean(post.α) # media a posteriori de α

begin
    # densidad a posteriori de α
    histogram(post.α, color = :blue, label = "densidad a posteriori", normalize = true)
    xaxis!(L"α")
    yaxis!(L"p(α|x)")
    title!("Histograma de las simulaciones a posteriori de α")
    vline!([αpost], lw = 4, color = :red, label = "media a posteriori round($(round(αpost; digits=3)))")
end
begin
    # densidad a posteriori de β
    histogram(post.β, color = :green, label = "densidad a posteriori", normalize = true)
    xaxis!(L"β")
    yaxis!(L"p(β|x)")
    title!("Histograma de las simulaciones a posteriori de β")
    vline!([βpost], lw = 4, color = :red, label = "media a posteriori round($(round(βpost; digits=3)))")
end


# Función predictiva a posteriori
function predictiva(simα::Vector, simβ::Vector, x::Real)
    m = length(simα)
    xsim = 0
    for i ∈ 1:m
        xsim += pdf(Beta(simα[i],simβ[i]), x)
    end
    return (1/m)*xsim 
    
end

x = collect(range(0.001, 1.0; length=1000))
y = [predictiva(post.α, post.β, xi) for xi in x]
# Calculamos la media de la predictiva (Estimación puntual)
ϕ(x) = x * predictiva(post.α, post.β, x)
mediax, error = quadgk(ϕ, 0, 1)

# Gráfico de la función de densidad de la predictiva a posteriori
begin
    xpost = mean(y)
    plot(x, y, label= "Densidad predictiva a posteriori", xlabel=L"x$_{(n+1)}$", ylabel=L"p(x$_{(n+1)}|x_{obs})$", lw=2, title="Predictiva a posteriori", color=:green)
    vline!([mediax], lw = 4, color = :red, label = "media a predictiva round($(round(mediax; digits=3)))")
end


i(x) = predictiva(post.α, post.β, x)
suma, error1 = quadgk(i, 0, 1)