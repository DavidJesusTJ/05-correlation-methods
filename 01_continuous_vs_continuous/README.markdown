# <center><b> Correlaciones de Variables Continuas vs Continuas </b></center>

---

## Correlación de Pearson

### Descripción:
La **correlación de Pearson** es una medida de asociación lineal entre dos variables cuantitativas continuas. Evalúa el grado en que un cambio en una variable se asocia con un cambio proporcional en otra. Su valor oscila entre -1 y 1:  
- 1 indica una correlación positiva perfecta (ambas variables aumentan juntas).  
- -1 indica una correlación negativa perfecta (una variable aumenta mientras la otra disminuye).  
- 0 indica ausencia de relación lineal.  

**Limitaciones:**  
- Solo detecta relaciones lineales, no captura asociaciones no lineales.  
- Sensible a valores extremos (outliers), que pueden distorsionar la correlación.  
- No implica causalidad.

### Formulación:
La correlación de Pearson se calcula como:

$$
r = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2} \sqrt{\sum_{i=1}^{n} (Y_i - \bar{Y})^2}}
$$

Donde:  
- $X_i$, $Y_i$ = valores individuales de las variables X e Y  
- $\bar{X}$, $\bar{Y}$ = medias de X e Y  
- $n$ = número de observaciones  
- $r$ = coeficiente de correlación de Pearson

### Supuestos:
- Las variables son **cuantitativas continuas**.  
- Relación **lineal** entre las variables.  
- **Normalidad bivariada** de las variables (no estrictamente necesaria para estimación, pero sí para inferencia y p-valor).  
- **Ausencia de outliers** significativos que puedan sesgar la medida.

### Interpretación:
- $r$ cercano a 1: fuerte correlación positiva  
- $r$ cercano a -1: fuerte correlación negativa  
- $r$ cercano a 0: no hay relación lineal significativa  
- El **p-valor** asociado indica si la correlación observada es estadísticamente significativa frente a la hipótesis nula de $r = 0$.  
- Siempre se debe complementar con un **diagrama de dispersión** para verificar la linealidad y posibles outliers.

---

## Correlación de Spearman

### Descripción:
La **correlación de Spearman** es una medida no paramétrica que evalúa la **relación monotónica** entre dos variables cuantitativas o ordinales. No requiere que la relación sea lineal ni que las variables sigan una distribución normal. Es útil cuando los datos contienen outliers o no cumplen los supuestos de Pearson.  
- Su valor también oscila entre -1 y 1:  
  - 1 indica una relación monotónica positiva perfecta  
  - -1 indica una relación monotónica negativa perfecta  
  - 0 indica ausencia de relación monotónica  

**Limitaciones:**  
- Solo detecta relaciones monotónicas, no captura patrones complejos no monotónicos.  
- Menos sensible a cambios en la magnitud de los valores que Pearson.

### Formulación:
Se basa en los **rangos** de las variables:

$$
\rho = 1 - \frac{6 \sum_{i=1}^{n} d_i^2}{n(n^2 - 1)}
$$

Donde:  
- $d_i = R(X_i) - R(Y_i)$ es la diferencia entre los rangos de X e Y para la observación $i$  
- $R(X_i)$, $R(Y_i)$ = rangos de las variables  
- $n$ = número de observaciones  
- $\rho$ = coeficiente de correlación de Spearman

### Supuestos:
- Las variables son **ordinales, intervalos o de razón** (cuantitativas).  
- La relación entre las variables es **monotónica** (no necesariamente lineal).  
- Puede manejar **outliers** de manera más robusta que Pearson.  

### Interpretación:
- $\rho$ cercano a 1: fuerte relación monotónica positiva  
- $\rho$ cercano a -1: fuerte relación monotónica negativa  
- $\rho$ cercano a 0: ausencia de relación monotónica  
- El **p-valor** indica si la relación observada es significativa frente a la hipótesis nula de $\rho = 0$.  
- Es recomendable graficar un **scatter plot con los rangos** para visualizar la monotonicidad.

---

## Correlación de Kendall

### Descripción:
La **correlación de Kendall** es una medida no paramétrica que evalúa la **asociación ordinal** entre dos variables. Al igual que Spearman, mide relaciones monotónicas, pero se basa en el **conteo de pares concordantes y discordantes**, lo que la hace más robusta frente a outliers y datos con empates.  

Existen dos variantes principales:  
- **Tau-B**: ajusta la correlación para empates en ambos ejes, ideal cuando las variables tienen empates frecuentes y tamaños similares de categorías.  
- **Tau-C**: se ajusta principalmente para **tablas rectangulares**, donde el número de categorías en X e Y difiere notablemente; útil cuando las variables tienen distinto número de niveles.  

- Valores de Kendall Tau también oscilan entre -1 y 1:  
  - 1 indica todos los pares concordantes  
  - -1 indica todos los pares discordantes  
  - 0 indica ausencia de asociación monotónica

### Formulación:
Sea $n$ el número de observaciones y consideremos todos los pares posibles $(i, j)$:

$$
\tau = \frac{C - D}{\sqrt{(C + D + T_x)(C + D + T_y)}}
$$

Donde:  
- $C$ = número de pares concordantes  
- $D$ = número de pares discordantes  
- $T_x$ = número de empates solo en X  
- $T_y$ = número de empates solo en Y  

- **Tau-B** utiliza la fórmula anterior incluyendo $T_x$ y $T_y$.  
- **Tau-C** ajusta el denominador considerando el tamaño de la tabla de contingencia (útil si las categorías no son iguales en X e Y).

### Supuestos:
- Las variables son **ordinales o cuantitativas** (rango o razón).  
- Relación **monotónica**.  
- Puede manejar **empates** entre valores (Tau-B más ajustado que Tau-C en tablas cuadradas).  

### Interpretación:
- $\tau$ cercano a 1: todos los pares son concordantes → fuerte relación positiva  
- $\tau$ cercano a -1: todos los pares son discordantes → fuerte relación negativa  
- $\tau$ cercano a 0: ausencia de relación monotónica  
- El **p-valor** indica significancia frente a la hipótesis nula de $\tau = 0$.  
- Tau-B es preferible si hay muchos empates y las variables tienen similar número de categorías, mientras que Tau-C es mejor si la tabla es rectangular (distinto número de categorías en X e Y).

---

## Correlación de Distancia (dCor)

### Descripción:
La **correlación de distancia (dCor, Distance Correlation)** mide la **dependencia entre dos variables** (o vectores de variables) y no se limita a relaciones lineales.  
A diferencia de Pearson, dCor puede detectar **relaciones no lineales** y es **0 si y solo si las variables son independientes**. Esto la hace especialmente útil en análisis exploratorio donde la relación no se asume lineal.  

- Valores van de 0 a 1:  
  - 0 indica independencia total  
  - 1 indica dependencia total (perfecta)  

### Formulación:
Sea $X = \{x_1, ..., x_n\}$ y $Y = \{y_1, ..., y_n\}$:  

1. Se calculan las matrices de distancias euclidianas:  
$$
A_{ij} = \|x_i - x_j\|, \quad B_{ij} = \|y_i - y_j\|
$$

2. Se centran las matrices:  
$$
\tilde{A}_{ij} = A_{ij} - \bar{A}_{i\cdot} - \bar{A}_{\cdot j} + \bar{A}_{\cdot\cdot}, \quad 
\tilde{B}_{ij} = B_{ij} - \bar{B}_{i\cdot} - \bar{B}_{\cdot j} + \bar{B}_{\cdot\cdot}
$$

3. La **correlación de distancia** se define como:  
$$
dCor(X, Y) = \frac{\text{dCov}(X, Y)}{\sqrt{\text{dCov}(X, X) \cdot \text{dCov}(Y, Y)}}
$$

donde la **covarianza de distancia** es:  
$$
\text{dCov}(X, Y) = \frac{1}{n^2} \sum_{i,j} \tilde{A}_{ij} \tilde{B}_{ij}
$$

### Supuestos:
- No requiere normalidad ni linealidad.  
- Variables continuas o multivariadas.  
- Independencia entre observaciones (cada fila de datos es una observación distinta).  

### Interpretación:
- **dCor cercano a 0** → variables aproximadamente independientes  
- **dCor cercano a 1** → fuerte dependencia  
- Detecta relaciones lineales y no lineales, por lo que es más general que Pearson o Spearman.  
- Se puede acompañar de un **test de significancia** mediante permutaciones para evaluar si la dependencia observada es estadísticamente significativa.

---

## Correlación de Hoeffding’s D

### Descripción:
La **correlación de Hoeffding's D** mide la **dependencia general entre dos variables continuas** sin asumir linealidad ni monotonicidad.  
Es particularmente útil para detectar relaciones complejas que no serían captadas por Pearson o Spearman.  

- Valores típicos:  
  - 0 indica independencia  
  - Valores positivos indican dependencia, sin un límite superior estricto de 1 (aunque suele estar acotado entre 0 y 0.5 para muestras grandes).  

**Limitaciones:**  
- Sensible a tamaño de muestra: con muestras muy pequeñas puede no detectar dependencia.  
- Computacionalmente más costosa que Pearson o Spearman.  

### Formulación:
Sea un conjunto de pares de observaciones $(X_i, Y_i)$, $i = 1, ..., n$. Definimos:  

1. Para cada par $(i,j)$, indicamos si $X_i < X_j$ y $Y_i < Y_j$.  
2. Se construyen frecuencias relativas de concordancia y discordancia en todos los pares.  

La estadística se calcula como:  
$$
D = \frac{30}{n(n-1)(n-2)(n-3)(n-4)} \sum (Q - \text{correcciones})
$$

donde $Q$ involucra conteos de pares y tripletas concordantes y discordantes.  

> Nota: La fórmula exacta es más extensa y compleja, y se suele calcular mediante librerías estadísticas.  

### Supuestos:
- Variables continuas.  
- Observaciones independientes.  
- No requiere normalidad ni linealidad.  

### Interpretación:
- **D ≈ 0** → las variables son aproximadamente independientes  
- **D > 0** → existe dependencia  
- No distingue la dirección de la relación, solo su fuerza general.  
- Se puede usar con un **test de significancia** para evaluar si la dependencia observada es estadísticamente significativa.

---

## Correlación de Blomqvist’s Beta

### Descripción:
La **correlación de Blomqvist's Beta** (también llamada mediana de correlación) mide la **dependencia entre dos variables continuas o discretas**, centrándose en la **mediana conjunta**.  
Es un estadístico robusto frente a valores extremos y útil cuando se quiere capturar la asociación central entre variables.  

- Valores típicos:  
  - Varía entre -1 y 1.  
  - 0 indica independencia alrededor de la mediana.  

**Limitaciones:**  
- No captura relaciones extremas fuera de la mediana.  
- Menos sensible a dependencias no centrales.  

### Formulación:
Sea $(X_i, Y_i)$ un conjunto de observaciones, y sea $m_X$ y $m_Y$ las medianas de $X$ y $Y$ respectivamente. Entonces:  

$$
\beta = P[(X - m_X)(Y - m_Y) > 0] - P[(X - m_X)(Y - m_Y) < 0]
$$

donde:  
- $P[(X - m_X)(Y - m_Y) > 0]$ es la probabilidad de que ambos estén en el mismo lado de sus medianas (concordancia alrededor de la mediana).  
- $P[(X - m_X)(Y - m_Y) < 0]$ es la probabilidad de discordancia alrededor de la mediana.  

### Supuestos:
- Variables continuas o discretas ordinales.  
- Observaciones independientes.  
- Se enfoca en la mediana; no requiere distribución normal ni linealidad.  

### Interpretación:
- **Beta ≈ 0** → independencia alrededor de la mediana  
- **Beta > 0** → concordancia central entre variables  
- **Beta < 0** → discordancia central  
- Robustez frente a valores extremos lo hace útil en presencia de outliers.

---

## Coeficiente de Información Máxima (MIC)

### Descripción:
El **Maximal Information Coefficient (MIC)** es una medida de dependencia **no lineal y general** entre dos variables continuas o discretas.  
Captura **cualquier tipo de relación funcional**, no solo lineal, y es simétrica: MIC(X, Y) = MIC(Y, X).  

- Valores típicos: entre 0 y 1.  
- MIC ≈ 0 indica independencia; MIC ≈ 1 indica fuerte dependencia funcional.  

**Limitaciones:**  
- Requiere un tamaño de muestra moderado para ser confiable.  
- Computacionalmente más costoso que Pearson o Spearman.  
- No indica la dirección de la relación, solo fuerza de dependencia.  

### Formulación:
MIC se basa en **información mutua normalizada** en diferentes discretizaciones de la variable:

$$
MIC(X, Y) = \max_{x\text{-bins}, y\text{-bins}} \frac{I(X_b; Y_b)}{\log_2 \min(x\text{-bins}, y\text{-bins})}
$$

donde:  
- $X_b, Y_b$ = versiones discretizadas de $X$ y $Y$ en una cuadrícula de tamaño variable.  
- $I(X_b; Y_b)$ = información mutua de las variables discretizadas.  
- La normalización divide por $\log_2 \min(\text{número de bins de X}, \text{número de bins de Y})$ para acotar MIC entre 0 y 1.  

### Supuestos:
- Variables continuas o discretas ordinales.  
- Observaciones independientes.  
- No requiere linealidad ni normalidad.  

### Interpretación:
- **MIC ≈ 0** → independencia entre variables.  
- **MIC ≈ 1** → fuerte dependencia funcional (lineal o no lineal).  
- Se usa principalmente para **descubrir asociaciones desconocidas o complejas**.

---

## Biweight Midcorrelation

### Descripción:
La **Biweight Midcorrelation (bicor)** es una correlación robusta que reduce el efecto de **outliers extremos**.  
A diferencia de Pearson, que se ve muy afectado por valores atípicos, bicor aplica un **peso decreciente** a observaciones alejadas del centro de la distribución, preservando la sensibilidad a la relación lineal central.

- Valores típicos: entre -1 y 1, al igual que Pearson.  
- Muy útil cuando hay datos ruidosos o distribuciones con colas pesadas.  

**Limitaciones:**  
- Menos eficiente que Pearson si los datos no tienen outliers.  
- No detecta relaciones no lineales.  

### Formulación:
Se basa en el **Biweight midcovariance**:

$$
u_i = \frac{x_i - \text{mediana}(X)}{9 \cdot \text{MAD}(X)}, \quad v_i = \frac{y_i - \text{mediana}(Y)}{9 \cdot \text{MAD}(Y)}
$$

$$
w_i = (1 - u_i^2)^2 \cdot \mathbf{1}_{|u_i| < 1}, \quad z_i = (1 - v_i^2)^2 \cdot \mathbf{1}_{|v_i| < 1}
$$

$$
\text{bicor}(X, Y) = \frac{\sum_i w_i z_i (x_i - \text{mediana}(X))(y_i - \text{mediana}(Y))}{\sqrt{\sum_i w_i^2 (x_i - \text{mediana}(X))^2 \sum_i z_i^2 (y_i - \text{mediana}(Y))^2}}
$$

donde:  
- $MAD(X)$ = mediana de las desviaciones absolutas de $X$.  
- $w_i, z_i$ = pesos que reducen la influencia de observaciones extremas.  

### Supuestos:
- Variables continuas.  
- Observaciones independientes.  
- Robustez a outliers; no requiere normalidad.  

### Interpretación:
- **bicor ≈ 1** → fuerte correlación positiva robusta.  
- **bicor ≈ -1** → fuerte correlación negativa robusta.  
- **bicor ≈ 0** → ausencia de relación lineal central.

---

## Correlación Winsorized

### Descripción:
La **correlación Winsorized** es una versión robusta de la correlación de Pearson que **reduce el efecto de valores extremos**.  
En lugar de eliminar outliers, los valores en los extremos se **recortan o "Winsorizan"** a ciertos percentiles (por ejemplo, 5% y 95%), preservando la estructura general de los datos.

- Útil cuando hay datos con outliers que podrían distorsionar Pearson.  
- Mantiene una interpretación similar a Pearson.  

**Limitaciones:**  
- La elección de los percentiles es arbitraria y puede afectar el resultado.  
- No detecta relaciones no lineales.  

### Formulación:
1. Definir el **percentil α** (por ejemplo 5%) para Winsorización.  
2. Reemplazar valores por debajo del percentil α por el valor del percentil α y valores por encima del percentil (1-α) por el percentil (1-α).  
3. Calcular la correlación de Pearson sobre los datos Winsorizados:

$$
\text{win\_cor}(X, Y) = \frac{\text{Cov}(X_w, Y_w)}{\sigma_{X_w} \sigma_{Y_w}}
$$

donde:  
- $X_w, Y_w$ = variables Winsorizadas.  
- $\text{Cov}$ = covarianza.  
- $\sigma_{X_w}, \sigma_{Y_w}$ = desviaciones estándar de los datos Winsorizados.  

### Supuestos:
- Variables continuas.  
- Observaciones independientes.  
- Robustez a outliers; no requiere normalidad.  

### Interpretación:
- Valores cercanos a **1** → fuerte correlación positiva tras recorte de extremos.  
- Valores cercanos a **-1** → fuerte correlación negativa robusta.  
- Valores cercanos a **0** → ausencia de relación lineal central.

---

## Correlación Parcial

### Descripción:
La **correlación parcial** mide la relación lineal entre dos variables **mientras se controla o ajusta por el efecto de una o más variables adicionales**.  
Es útil para aislar la relación directa entre dos variables eliminando la influencia de otras variables que podrían confundir la asociación.

- Permite identificar correlaciones "reales" entre dos variables cuando hay variables de confusión.  
- Se aplica tanto en análisis exploratorio como en modelos multivariantes.  

**Limitaciones:**  
- Asume linealidad entre variables ajustadas.  
- Sensible a colinealidad alta entre variables controladas.  
- No captura relaciones no lineales.  

### Formulación:
Sea $X$, $Y$ las variables de interés y $Z$ la variable controlada:

1. Ajustar $X$ sobre $Z$: obtener residuales $R_X = X - \hat{X}_Z$.  
2. Ajustar $Y$ sobre $Z$: obtener residuales $R_Y = Y - \hat{Y}_Z$.  
3. Calcular correlación de Pearson entre los residuales:

$$
r_{XY \cdot Z} = \frac{\text{Cov}(R_X, R_Y)}{\sigma_{R_X} \sigma_{R_Y}}
$$

donde:  
- $r_{XY \cdot Z}$ = correlación parcial entre $X$ y $Y$ controlando $Z$.  
- $R_X, R_Y$ = residuos después de ajustar por $Z$.  
- $\text{Cov}$ = covarianza, $\sigma$ = desviación estándar.  

### Supuestos:
- Variables continuas.  
- Linealidad entre variables ajustadas.  
- Observaciones independientes.  
- Homocedasticidad de los residuos.  

### Interpretación:
- Valores cercanos a **1** → fuerte relación positiva directa entre $X$ y $Y$ tras controlar $Z$.  
- Valores cercanos a **-1** → fuerte relación negativa directa entre $X$ y $Y$.  
- Valores cercanos a **0** → no existe correlación directa entre $X$ y $Y$ después de ajustar por $Z$.  
- P-valor asociado indica si la correlación parcial es estadísticamente significativa.

---

## Copula-based Correlation

### Descripción:
La **correlación basada en copulas** mide la dependencia entre dos variables considerando su **estructura conjunta y marginals**, sin asumir linealidad ni distribuciones normales.  
Las copulas permiten separar la dependencia de la distribución marginal de cada variable, capturando relaciones **lineales y no lineales**, incluyendo colas de distribución.

- Muy útil en finanzas, seguros y análisis de riesgos donde las relaciones extremas son importantes.  
- Permite trabajar con variables de diferentes escalas y distribuciones.  

**Limitaciones:**  
- La elección del tipo de copula (Clayton, Gumbel, Frank, Gaussian, etc.) puede afectar los resultados.  
- Requiere muestras relativamente grandes para estimaciones precisas.  
- No proporciona un valor de correlación tradicional entre -1 y 1, sino medidas de dependencia como Kendall’s tau o Spearman rho derivados de la copula.

### Formulación:
Sea $F_X$ y $F_Y$ las distribuciones marginales de $X$ y $Y$, y $C$ la copula que describe su dependencia:

$$
C(u, v) = P(F_X(X) \leq u, F_Y(Y) \leq v), \quad u, v \in [0,1]
$$

A partir de la copula se puede obtener la correlación de dependencia como **Kendall’s tau**:

$$
\tau = 4 \int_0^1 \int_0^1 C(u,v) \, dC(u,v) - 1
$$

donde:  
- $u = F_X(X)$, $v = F_Y(Y)$ (transformación a escala uniforme).  
- $C(u,v)$ = función copula que modela la dependencia conjunta.  
- $\tau$ = medida de dependencia derivada de la copula, equivalente a Kendall’s tau.

### Supuestos:
- Variables continuas (aunque existen copulas para discretas).  
- Observaciones independientes.  
- La copula seleccionada refleja adecuadamente la dependencia entre variables.  

### Interpretación:
- $\tau$ cercano a **1** → fuerte dependencia positiva.  
- $\tau$ cercano a **-1** → fuerte dependencia negativa.  
- $\tau$ cercano a **0** → casi independencia.  
- A diferencia de Pearson, la copula captura **relaciones no lineales y dependencias en colas**.

---

## Quadrant Correlation (Q)

### Descripción:
La **Correlación de Cuadrantes (Quadrant Correlation, Q)** es una medida robusta de asociación entre dos variables continuas.  
- Evalúa la dependencia comparando la posición de los puntos respecto a la **mediana de cada variable** y contando cuántos caen en el mismo cuadrante.  
- Es menos sensible a **outliers** que la correlación de Pearson.  
- Se usa cuando se sospecha que los datos tienen valores extremos que podrían distorsionar medidas lineales tradicionales.

**Limitaciones:**  
- Solo captura la **dirección general de la asociación**, no la fuerza exacta como Pearson.  
- Puede ser menos eficiente en muestras pequeñas.

### Formulación:
Sea $(x_i, y_i)$ cada observación, y $med(X)$ y $med(Y)$ las medianas de $X$ y $Y$. Definimos:

$$
Q = \frac{(n_{++} + n_{--}) - (n_{+-} + n_{-+})}{n}
$$

donde:  
- $n_{++}$ = cantidad de puntos donde $x_i > med(X)$ y $y_i > med(Y)$  
- $n_{--}$ = cantidad de puntos donde $x_i < med(X)$ y $y_i < med(Y)$  
- $n_{+-}$ = cantidad de puntos donde $x_i > med(X)$ y $y_i < med(Y)$  
- $n_{-+}$ = cantidad de puntos donde $x_i < med(X)$ y $y_i > med(Y)$  
- $n$ = número total de observaciones

### Supuestos:
- Variables continuas o al menos ordinales.  
- Observaciones independientes.  
- Mediana bien definida (no demasiados empates).

### Interpretación:
- $Q \approx 1$ → fuerte asociación positiva.  
- $Q \approx -1$ → fuerte asociación negativa.  
- $Q \approx 0$ → independencia o ausencia de tendencia lineal.  
- Al ser robusta, no se ve afectada por outliers extremos.

---

## Correlación del Porcentaje de Curvatura

### Descripción:
La **Correlación del Porcentaje de Curvatura (Percentage Bend Correlation, PBC)** es una medida robusta de asociación entre dos variables continuas.  
- Su objetivo es reducir el impacto de **outliers** al "doblar" los valores extremos hacia la mediana.  
- Es útil cuando se desea una estimación de correlación similar a Pearson pero menos sensible a observaciones atípicas.  
- Limita la influencia de valores extremos mediante un parámetro de **bend (corte)** que determina qué proporción de los datos se ajusta.

**Limitaciones:**  
- No captura completamente relaciones muy complejas o no lineales.  
- Depende del parámetro de bend, que debe seleccionarse con cuidado.

### Formulación:
Sea $X$ y $Y$ las variables y $c$ el **porcentaje de bend** (por ejemplo, 0.2). Se define un vector robusto centrado:

$$
x_i^* = 
\begin{cases} 
med(X) + c \cdot (x_i - med(X)), & \text{si } |x_i - med(X)| > c \cdot MAD(X) \\
x_i, & \text{en otro caso}
\end{cases}
$$

$$
y_i^* = 
\begin{cases} 
med(Y) + c \cdot (y_i - med(Y)), & \text{si } |y_i - med(Y)| > c \cdot MAD(Y) \\
y_i, & \text{en otro caso}
\end{cases}
$$

Luego la correlación se calcula como Pearson entre $x^*$ y $y^*$:

$$
r_{PBC} = \frac{\sum_i (x_i^* - \bar{x^*})(y_i^* - \bar{y^*})}{\sqrt{\sum_i (x_i^* - \bar{x^*})^2 \sum_i (y_i^* - \bar{y^*})^2}}
$$

donde $MAD(X)$ es la **desviación absoluta mediana** de $X$.

### Supuestos:
- Variables continuas o al menos ordinales.  
- Observaciones independientes.  
- Presencia posible de outliers.

### Interpretación:
- $r_{PBC} \approx 1$ → fuerte asociación positiva robusta.  
- $r_{PBC} \approx -1$ → fuerte asociación negativa robusta.  
- $r_{PBC} \approx 0$ → ausencia de tendencia lineal, robusta frente a outliers.

---

## Correlación Pi de Shepherd

### Descripción:
La **Correlación Pi de Shepherd (Shepherd’s Pi Correlation)** es una medida robusta de asociación que evalúa la relación entre dos variables de manera resistente a **outliers y datos atípicos**.  
- Se basa en la **distancia entre pares de observaciones** y ajusta su influencia para valores extremos.  
- Es útil cuando se quiere capturar la tendencia general de la relación sin que valores extremos distorsionen el resultado.  
- Limita la influencia de puntos atípicos mediante un **corte (trim)** que excluye los valores más extremos en los cálculos.

**Limitaciones:**  
- No es adecuada para datos categóricos.  
- Puede ser sensible al parámetro de trim, que define la proporción de datos extremos a descartar.

### Formulación:
Sea $X$ y $Y$ los vectores de datos y $p$ la proporción de trimming (por ejemplo, 0.2).  
Se define el conjunto de observaciones centrales $\tilde{X}, \tilde{Y}$ descartando los valores extremos según $p$, y luego se calcula la correlación como:

$$
\pi = \frac{\sum_{i=1}^{n'} ( \tilde{x}_i - \bar{\tilde{x}} ) ( \tilde{y}_i - \bar{\tilde{y}} )}{\sqrt{\sum_{i=1}^{n'} ( \tilde{x}_i - \bar{\tilde{x}} )^2 \sum_{i=1}^{n'} ( \tilde{y}_i - \bar{\tilde{y}} )^2}}
$$

donde $n' = (1 - 2p)n$ es el número de observaciones consideradas tras el trimming.

### Supuestos:
- Variables continuas.  
- Observaciones independientes.  
- Puede manejar la presencia de outliers, aunque depende del parámetro de trimming.  

### Interpretación:
- $\pi \approx 1$ → fuerte asociación positiva robusta.  
- $\pi \approx -1$ → fuerte asociación negativa robusta.  
- $\pi \approx 0$ → ausencia de relación lineal, robusta frente a datos extremos.

---

## Correlación Omitida

### Descripción:
La **Correlación Omitida (Skipped Correlation)** es una medida robusta de asociación que busca estimar la correlación entre dos variables **ignorando los outliers**.  
- Se identifica y excluye un porcentaje de valores extremos antes de calcular la correlación, asegurando que los valores atípicos no distorsionen la estimación.  
- Es especialmente útil en datasets con presencia de valores aberrantes que podrían afectar a las correlaciones tradicionales como Pearson o Spearman.

**Limitaciones:**  
- La proporción de datos a omitir debe seleccionarse cuidadosamente.  
- No está diseñada para datos categóricos ni para variables con muchas repeticiones.

### Formulación:
Sea $X$ y $Y$ los vectores de datos y $k$ el número de observaciones a omitir (basado en outliers detectados).  
Se definen los subconjuntos $X', Y'$ excluyendo los outliers y luego se calcula la correlación de Pearson sobre el subconjunto:

$$
r_{skipped} = \frac{\sum_{i \in S} (x_i - \bar{x}') (y_i - \bar{y}')}{\sqrt{\sum_{i \in S} (x_i - \bar{x}')^2 \sum_{i \in S} (y_i - \bar{y}')^2}}
$$

donde $S$ es el conjunto de índices **no omitidos**, y $\bar{x}', \bar{y}'$ son las medias sobre $S$.

### Supuestos:
- Variables continuas.  
- Observaciones independientes.  
- La robustez depende de la detección correcta de outliers.  

### Interpretación:
- $r_{skipped} \approx 1$ → fuerte relación positiva robusta.  
- $r_{skipped} \approx -1$ → fuerte relación negativa robusta.  
- $r_{skipped} \approx 0$ → ausencia de relación lineal entre las variables tras omitir outliers.

---