import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import plotly.graph_objects as plotly_go
from scipy.optimize import line_search

st.set_page_config(
    page_title="OptiWeb Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ""
if 'optimization_results' not in st.session_state:
    st.session_state['optimization_results'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Inicio"

def prepare_functions(func_str, var_str):
    """
    Convierte la cadena de texto de la función en funciones evaluables de Python
    para f(x), el gradiente y la Hessiana usando cálculo simbólico (SymPy).
    """
    try:
        # Definir símbolos
        var_names = [v.strip() for v in var_str.split(',')]
        symbols = sp.symbols(var_names)
        
        # Parsear función
        f_sym = sp.sympify(func_str)
        
        # Calcular Gradiente de forma analítica
        grad_sym = [sp.diff(f_sym, var) for var in symbols]
        
        # Calcular Matriz Hessiana de forma analítica
        hessian_sym = [[sp.diff(g, var) for var in symbols] for g in grad_sym]
        
        # Convertir a funciones de numpy para evaluación rápida
        f_lamb = sp.lambdify(symbols, f_sym, 'numpy')
        grad_lamb = [sp.lambdify(symbols, g, 'numpy') for g in grad_sym]
        hess_lamb = [[sp.lambdify(symbols, h, 'numpy') for h in row] for row in hessian_sym]
        
        # Wrappers para aceptar un array (vector) x
        def f_val(x):
            return float(f_lamb(*x))
            
        def grad_val(x):
            return np.array([g(*x) for g in grad_lamb], dtype=float)
            
        def hess_val(x):
            return np.array([[h(*x) for h in row] for row in hess_lamb], dtype=float)
            
        return f_val, grad_val, hess_val, symbols, f_sym
    except Exception as e:
        raise ValueError(f"Error al procesar la función matemática: {e}. Asegúrate de usar sintaxis de Python (ej: x1**2).")

def optimize(f, grad, hess, x0, method, max_iter, tol, c1, c2):
    """
    Ejecuta el método de optimización seleccionado garantizando las condiciones de Wolfe.
    """
    xk = np.array(x0, dtype=float)
    history = []
    
    # Evaluar punto inicial
    fk = f(xk)
    gk = grad(xk)
    norm_gk = np.linalg.norm(gk)
    
    history.append({
        'k': 0, 'x': xk.copy(), 'f(x)': fk, '||grad f(x)||': norm_gk, 'alpha': None
    })
    
    pk = -gk # Dirección inicial por defecto
    
    for k in range(1, max_iter + 1):
        if norm_gk < tol:
            break
            
        # 1. Calcular dirección de descenso pk según el método seleccionado
        if method == "Gradiente":
            pk = -gk
        elif method == "Gradiente Conjugado (Fletcher-Reeves)":
            if k == 1:
                pk = -gk
            else:
                gk_prev = history[-2]['grad_raw']
                # Fórmula de Fletcher-Reeves para Beta
                beta = np.dot(gk, gk) / np.dot(gk_prev, gk_prev)
                pk = -gk + beta * history[-1]['p_raw']
        elif method == "Newton":
            Hk = hess(xk)
            try:
                # Resolver Hk * pk = -gk
                pk = np.linalg.solve(Hk, -gk)
            except np.linalg.LinAlgError:
                # Fallback: Si la Hessiana no es invertible o condicionada, usar descenso de gradiente
                pk = -gk
                
        # line_search de scipy implementa condiciones de Wolfe fuertes (Armijo + Curvatura)
        res = line_search(f, grad, xk, pk, gfk=gk, old_fval=fk, c1=c1, c2=c2)
        alpha = res[0]
        
        # Fallback si las condiciones de Wolfe fallan en encontrar un paso válido
        if alpha is None:
            alpha = 1e-4 
            
        # 3. Actualizar variables para la siguiente iteración
        xk = xk + alpha * pk
        fk = f(xk)
        gk = grad(xk)
        norm_gk = np.linalg.norm(gk)
        
        history.append({
            'k': k,
            'x': xk.copy(),
            'f(x)': fk,
            '||grad f(x)||': norm_gk,
            'alpha': alpha,
            'grad_raw': gk.copy(), 
            'p_raw': pk.copy()     
        })
        
    return history

def sidebar_navigation():
    st.sidebar.title("OptiWeb Solver")
    if st.session_state['logged_in']:
        st.sidebar.markdown(f"👤 **{st.session_state['user_name']}**")
        st.sidebar.markdown("---")
        
        # Botones de navegación
        pages = {
            "Configuración": "⚙️",
            "Gráficos": "📈",
            "Iteraciones": "📋"
        }
        
        for page, icon in pages.items():
            if st.sidebar.button(f"{icon} {page}", use_container_width=True):
                st.session_state['current_page'] = page
                
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['current_page'] = "Inicio"
            st.session_state['optimization_results'] = None
            st.rerun()

def page_inicio():
    st.markdown("<h1 style='text-align: center; color: #4f46e5; margin-top: 50px;'>OptiWeb Solver</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #64748b;'>Plataforma interactiva para encontrar mínimos de funciones.</p>", unsafe_allow_html=True)
    
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background-color: #f8fafc; padding: 2rem; border-radius: 10px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <h3 style="text-align: center; margin-bottom: 20px; color: #1e293b;">Acceso al Sistema</h3>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            user = st.text_input("Nombre del Investigador / Alumno", placeholder="Ingresa tu nombre para comenzar...")
            submit = st.form_submit_button("Ingresar", type="primary", use_container_width=True)
            
            if submit:
                if user.strip() == "":
                    st.error("Por favor, ingresa tu nombre para continuar.")
                else:
                    st.session_state['logged_in'] = True
                    st.session_state['user_name'] = user
                    st.session_state['current_page'] = "Configuración"
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

def page_configuracion():
    st.title("⚙️ Configuración del Problema")
    st.markdown("Define la función objetivo, el método a utilizar y los criterios de parada.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Función y Variables")
        # Función de Rosenbrock por defecto
        func_str = st.text_input("Función Objetivo f(x)", value="(x1 - 1)**2 + 100*(x2 - x1**2)**2", help="Usa notación de Python (ej: x1**2 para potencia, sp.exp(x1) para exponencial)")
        vars_str = st.text_input("Variables (separadas por coma)", value="x1, x2")
        
        st.subheader("2. Método y Punto Inicial")
        method = st.selectbox("Método de Optimización", ["Gradiente", "Gradiente Conjugado (Fletcher-Reeves)", "Newton"])
        x0_str = st.text_input("Punto de Partida (x0)", value="-1.2, 1.0", help="Valores separados por coma. Ej: -1.2, 1.0")
        
    with col2:
        st.subheader("3. Criterios de Parada")
        col_tol1, col_tol2 = st.columns(2)
        with col_tol1:
            max_iter = st.number_input("Iteraciones Máx.", min_value=1, max_value=10000, value=1000)
        with col_tol2:
            tol = st.number_input("Tolerancia (ε)", min_value=1e-12, max_value=0.1, value=1e-6, format="%e", help="Criterio de parada: ||∇f(x)|| < ε")
            
        st.subheader("4. Condiciones de Wolfe")
        st.markdown("*Parámetros para la búsqueda de línea*")
        c1 = st.slider("c1 (Condición de Armijo / Descenso Suficiente)", min_value=1e-5, max_value=0.1, value=1e-4, format="%f", help="Asegura que la función decrezca lo suficiente.")
        c2 = st.slider("c2 (Condición de Curvatura)", min_value=0.1, max_value=0.99, value=0.9, format="%f", help="Asegura pasos suficientemente largos.")
        
    st.markdown("---")
    if st.button("🚀 Ejecutar Optimización", type="primary", use_container_width=True):
        with st.spinner('Calculando gradientes, Hessiana y ejecutando el algoritmo...'):
            try:
                # Procesar datos de entrada
                x0 = [float(x.strip()) for x in x0_str.split(',')]
                f, grad, hess, syms, f_sym = prepare_functions(func_str, vars_str)
                
                if len(x0) != len(syms):
                    st.error(f"Error: El punto inicial debe tener {len(syms)} componentes (tienes {len(x0)}).")
                    return
                
                # Ejecutar
                history = optimize(f, grad, hess, x0, method, max_iter, tol, c1, c2)
                
                # Guardar en memoria
                st.session_state['optimization_results'] = {
                    'history': history,
                    'func_str': func_str,
                    'method': method,
                    'f_eval': f,
                    'vars': [str(s) for s in syms]
                }
                
                st.success("¡Optimización completada con éxito!")
                st.session_state['current_page'] = "Gráficos"
                st.rerun()
                
            except Exception as e:
                st.error(f"Ocurrió un error matemático o de sintaxis: {str(e)}")

def page_graficos():
    st.title("📈 Análisis Gráfico")
    
    if st.session_state['optimization_results'] is None:
        st.info("👋 Aún no hay datos para mostrar. Ve a **Configuración** y ejecuta una optimización primero.")
        return
        
    res = st.session_state['optimization_results']
    history = res['history']
    last_iter = history[-1]
    
    # Tarjetas de Resumen Rápido
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"**Punto Mínimo (x*)**\n\n[{', '.join([f'{x:.4f}' for x in last_iter['x']])}]")
    with col2:
        st.success(f"**Valor Mínimo f(x*)**\n\n{last_iter['f(x)']:e}")
    with col3:
        st.warning(f"**Iteraciones**\n\n{last_iter['k']}")
    with col4:
        st.error(f"**Error Final ||∇f||**\n\n{last_iter['||grad f(x)||']:e}")
    
    st.markdown("---")
    
    # Extraer datos para ploteo
    iters = [h['k'] for h in history]
    errors = [h['||grad f(x)||'] for h in history]
    
    path_x = [h['x'][0] for h in history]
    path_y = [h['x'][1] for h in history] if len(history[0]['x']) > 1 else [0]*len(history)
    path_z = [h['f(x)'] for h in history]
    
    col_graph1, col_graph2 = st.columns(2)
    
    # 1. Gráfico de Convergencia
    with col_graph1:
        st.subheader("Gráfico de Convergencia")
        st.markdown("Muestra la caída del error (Norma del Gradiente) iteración tras iteración.")
        
        fig_conv = plotly_go.Figure()
        fig_conv.add_trace(plotly_go.Scatter(x=iters, y=errors, mode='lines+markers', name='||∇f||', line=dict(color='#4f46e5', width=2), marker=dict(size=6)))
        fig_conv.update_layout(
            yaxis_type="log", 
            xaxis_title="Número de Iteración (k)", 
            yaxis_title="Log( ||∇f(x)|| )", 
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_conv, use_container_width=True)
        
    with col_graph2:
        st.subheader("Espacio de Soluciones")
        
        if len(history[0]['x']) == 2:
            st.markdown("Superficie de la función y trayectoria del algoritmo.")
            f_eval = res['f_eval']
            
            # Dinámicamente calcular límites del gráfico basados en la ruta recorrida
            margin_x = abs(max(path_x) - min(path_x)) * 0.5 + 0.5
            margin_y = abs(max(path_y) - min(path_y)) * 0.5 + 0.5
            x_min, x_max = min(path_x) - margin_x, max(path_x) + margin_x
            y_min, y_max = min(path_y) - margin_y, max(path_y) + margin_y
            
            # Malla de puntos
            x_grid = np.linspace(x_min, x_max, 50)
            y_grid = np.linspace(y_min, y_max, 50)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            # Evaluar la matriz Z
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    Z[i,j] = f_eval([X[i,j], Y[i,j]])
                    
            tab1, tab2 = st.tabs(["🗺️ Curvas de Nivel (Contorno)", "🧊 Superficie 3D"])
            
            with tab1:
                fig_cont = plotly_go.Figure(data=[plotly_go.Contour(z=Z, x=x_grid, y=y_grid, colorscale='Viridis', opacity=0.7)])
                fig_cont.add_trace(plotly_go.Scatter(x=path_x, y=path_y, mode='lines+markers', line=dict(color='red', width=2), marker=dict(size=6, color='red'), name='Ruta del Algoritmo'))
                fig_cont.update_layout(xaxis_title=res['vars'][0], yaxis_title=res['vars'][1], template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_cont, use_container_width=True)
                
            with tab2:
                fig_3d = plotly_go.Figure(data=[plotly_go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])
                fig_3d.add_trace(plotly_go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines+markers', line=dict(color='red', width=4), marker=dict(size=4, color='red'), name='Ruta'))
                fig_3d.update_layout(scene=dict(xaxis_title=res['vars'][0], yaxis_title=res['vars'][1], zaxis_title='f(x)'), margin=dict(l=0, r=0, b=0, t=0), template="plotly_white")
                st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("💡 Los gráficos de superficie y curvas de nivel solo se generan para funciones de exactamente 2 variables.")

def page_iteraciones():
    st.title("📋 Tabla de Iteraciones")
    
    if st.session_state['optimization_results'] is None:
        st.info("👋 Aún no hay datos para mostrar. Ve a **Configuración** y ejecuta una optimización primero.")
        return
        
    res = st.session_state['optimization_results']
    history = res['history']
    
    st.markdown(f"**Método utilizado:** `{res['method']}` | **Función:** `{res['func_str']}`")
    
    # Formatear los datos para mostrarlos limpios en un DataFrame de Pandas
    df_data = []
    for h in history:
        row = {
            'Iteración (k)': h['k'],
            'Punto Xk': f"[{', '.join([f'{x:.6f}' for x in h['x']])}]",
            'f(Xk)': f"{h['f(x)']:.8e}",
            '||∇f(Xk)||': f"{h['||grad f(x)||']:.8e}",
            'Paso (αk)': f"{h['alpha']:.6f}" if h['alpha'] is not None else "-"
        }
        df_data.append(row)
        
    df = pd.DataFrame(df_data)
    
    # Mostrar tabla interactiva (el usuario puede ordenar y hacer scroll)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        # Funcionalidad extra: Descargar como CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Tabla en CSV",
            data=csv,
            file_name='resultados_optimizacion.csv',
            mime='text/csv',
            type="primary"
        )
    with col2:
        st.caption("Puedes descargar estos datos para incluirlos en el informe de tu trabajo grupal.")

def main():
    # Enrutamiento de páginas
    if not st.session_state['logged_in']:
        page_inicio()
    else:
        sidebar_navigation()
        
        if st.session_state['current_page'] == "Inicio":
            # Si alguien fuerza la página de inicio estando logueado, lo mandamos a config
            st.session_state['current_page'] = "Configuración"
            st.rerun()
        elif st.session_state['current_page'] == "Configuración":
            page_configuracion()
        elif st.session_state['current_page'] == "Gráficos":
            page_graficos()
            