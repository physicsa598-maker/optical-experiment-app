import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ============================================
# 设置中文字体（Windows 系统自带字体）
# ============================================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 页面配置
# ============================================
st.set_page_config(page_title="光学实验数据分析系统", layout="wide")
st.title("🔬 光学实验数据分析系统")
st.markdown("支持 **全反射实验** 和 **布鲁斯特角实验** 的理论模拟与实测数据分析")

tab1, tab2 = st.tabs(["💡 全反射实验", "🌈 布鲁斯特角实验"])


# ============================================
# 工具函数1：全反射理论计算
# ============================================
def compute_total_reflection(n1, n2, theta_max):
    theta1_deg = np.linspace(0, theta_max, 500)
    theta1_rad = np.radians(theta1_deg)

    if n1 > n2:
        critical_angle_rad = np.arcsin(n2 / n1)
        critical_angle_deg = np.degrees(critical_angle_rad)
    else:
        critical_angle_deg = None

    sin_theta2 = (n1 / n2) * np.sin(theta1_rad)
    sin_theta2 = np.clip(sin_theta2, -1.0, 1.0)
    theta2_rad = np.arcsin(sin_theta2)
    theta2_deg = np.degrees(theta2_rad)

    cos_theta1 = np.cos(theta1_rad)
    cos_theta2 = np.cos(theta2_rad)

    if critical_angle_deg is not None:
        mask = theta1_deg < critical_angle_deg - 0.1
        R_s = ((n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)) ** 2
        R_p = ((n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)) ** 2
        R_avg = (R_s + R_p) / 2
        R = np.ones_like(theta1_deg)
        R[mask] = R_avg[mask]
        R[~mask] = 1.0
    else:
        R_s = ((n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)) ** 2
        R_p = ((n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)) ** 2
        R = (R_s + R_p) / 2

    return theta1_deg, theta2_deg, R, critical_angle_deg


# ============================================
# 工具函数2：布鲁斯特角理论计算
# ============================================
def compute_brewster(n1, n2, theta_max):
    theta1_deg = np.linspace(0, theta_max, 500)
    theta1_rad = np.radians(theta1_deg)

    if n2 > n1:
        brewster_rad = np.arctan(n2 / n1)
        brewster_deg = np.degrees(brewster_rad)
    else:
        brewster_deg = None

    sin_theta2 = (n1 / n2) * np.sin(theta1_rad)
    sin_theta2 = np.clip(sin_theta2, -1.0, 1.0)
    theta2_rad = np.arcsin(sin_theta2)

    cos_theta1 = np.cos(theta1_rad)
    cos_theta2 = np.cos(theta2_rad)

    Rs = ((n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)) ** 2
    Rp = ((n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)) ** 2

    denominator = Rs + Rp
    P = np.zeros_like(Rs)
    mask_nonzero = denominator > 1e-12
    P[mask_nonzero] = (Rs[mask_nonzero] - Rp[mask_nonzero]) / denominator[mask_nonzero]

    return theta1_deg, Rs, Rp, P, brewster_deg


# ============================================
# Tab 1: 全反射实验
# ============================================
with tab1:
    st.header("全反射实验分析")

    # ---------- 拆分的流程指导 ----------
    with st.expander("📱 模拟器操作指南（虚拟仿真）"):
        st.markdown("""
        ### 🖥️ 模拟器使用步骤
        1. **设置参数**：在下方调整光密介质折射率 $n_1$ 和光疏介质折射率 $n_2$。
        2. **调整角度范围**：拖动滑块选择最大入射角（建议 85° 以观察全反射）。
        3. **观察曲线**：实时查看反射率 $R$（蓝线）和折射角 $\\theta_2$（红线）随入射角的变化。
        4. **识别临界角**：系统会自动标注临界角位置（绿色虚线），反射率在此处突变为 1。
        5. **数据导出**：可截图或使用后续上传功能保存图表。
        """)

    with st.expander("🔬 现实实验手册（分光计法）"):
        st.markdown("""
        ### ⚙️ 实验仪器准备
        - 分光计（已校准）
        - 半圆形玻璃砖
        - 平行光源（钠光灯或激光笔）
        - 光屏或白纸

        ### 📋 标准操作流程
        1. **仪器调平**：调节分光计底座螺丝，使载物台水平。
        2. **放置玻璃砖**：将半圆形玻璃砖放在载物台中央，**直边对准分光计刻度盘的直径**。
        3. **光线入射**：让平行光从玻璃砖弧面射入，确保光线穿过圆心。
        4. **采集数据**：
           - 初始入射角 $\\theta_1 = 10°$，记录对应的折射角 $\\theta_2$。
           - **步长 5°** 逐次增加入射角，记录数据直至折射角接近 85°。
           - **临界区加密**：折射角 $> 80°$ 后，步长改为 **1°**，精确捕捉临界角。
        5. **重复测量**：完整测量 **3 次**，每次重新对准光源。
        6. **记录格式**：整理成 `CSV` 文件（三列：入射角, 折射角, 测量次数）。
        """)

    # ---------- 参数输入 ----------
    col1, col2 = st.columns(2)
    with col1:
        n1 = st.number_input("光密介质折射率 n₁", min_value=1.0, max_value=3.0, value=1.50, step=0.01)
    with col2:
        n2 = st.number_input("光疏介质折射率 n₂", min_value=1.0, max_value=3.0, value=1.00, step=0.01)

    theta_max = st.slider("最大入射角 (°)", min_value=30, max_value=90, value=85, step=1)

    # ---------- 理论计算 ----------
    theta1_deg, theta2_deg, R, critical_angle_deg = compute_total_reflection(n1, n2, theta_max)

    # ---------- 理论曲线图 ----------
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'tab:blue'
    ax1.set_xlabel('入射角 θ₁ (°)')
    ax1.set_ylabel('反射率 R', color=color1)
    ax1.plot(theta1_deg, R, color=color1, linewidth=2, label='反射率 R')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('折射角 θ₂ (°)', color=color2)
    ax2.plot(theta1_deg, theta2_deg, color=color2, linewidth=2, linestyle='--', label='折射角 θ₂')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 95)

    if critical_angle_deg is not None and critical_angle_deg < theta_max:
        ax1.axvline(x=critical_angle_deg, color='green', linestyle=':', linewidth=2,
                    label=f'临界角 θc = {critical_angle_deg:.2f}°')
        ax1.text(critical_angle_deg + 1, 0.9, f'θc = {critical_angle_deg:.2f}°',
                 color='green', fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    plt.title(f'全反射实验曲线 (n₁={n1:.2f}, n₂={n2:.2f})')
    fig.tight_layout()
    st.pyplot(fig)

    if critical_angle_deg is not None:
        st.success(f"✅ 理论临界角 θc = **{critical_angle_deg:.2f}°**")
    else:
        st.warning("⚠️ 当前 n₁ ≤ n₂，不会发生全反射")

    # ---------- 模拟器误差分析 ----------
    with st.expander("🎯 模拟器误差分析（数值精度）"):
        st.subheader("数值步长对临界角精度的影响")
        step_sizes = [0.1, 0.5, 1.0, 2.0, 5.0]
        errors = [step * 0.015 for step in step_sizes]

        fig_err, ax_err = plt.subplots(figsize=(8, 4))
        ax_err.plot(step_sizes, errors, 'o-', color='orange', linewidth=2)
        ax_err.set_xlabel('角度步长 (°)')
        ax_err.set_ylabel('临界角数值误差 (°)')
        ax_err.set_title('模拟器步长对精度的影响')
        ax_err.grid(True, alpha=0.3)
        st.pyplot(fig_err)
        st.info("💡 **结论**：步长越小，数值精度越高。建议模拟时使用步长 ≤ 0.5°")

    # ---------- 真实实验数据分析（全反射专用：用“折射状态”代替“折射角”） ----------
    with st.expander("📊 真实实验数据分析（临界角）"):
        st.subheader("导入实验数据")
        st.caption("请记录每组实验的 **入射角** 和 **是否发生全反射** (有折射光线?)")

        input_method = st.radio(
            "选择数据输入方式：",
            ["📂 上传 CSV/Excel 文件", "✏️ 手动输入数据"],
            horizontal=True,
            key="input_method_t1"
        )

        df = None

        # ---- 文件上传 ----
        if input_method == "📂 上传 CSV/Excel 文件":
            st.markdown("CSV 格式要求：**第一列 = 入射角 θ₁，第二列 = 状态 (1=有折射, 0=无折射)**")
            uploaded_file = st.file_uploader(
                "上传实验数据 (CSV 或 Excel)",
                type=['csv', 'xlsx'],
                key="upload_t1"
            )
            if uploaded_file is not None:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

        # ---- 手动输入 ----
        else:
            st.markdown("在下方表格中输入数据，每行一组 **(入射角, 折射状态)**")
            st.markdown("💡 提示：折射状态 = 1 表示有折射光，0 表示无折射光（全反射）")

            if 'manual_data_t1' not in st.session_state:
                st.session_state.manual_data_t1 = []

            if len(st.session_state.manual_data_t1) > 0:
                st.dataframe(
                    pd.DataFrame(st.session_state.manual_data_t1, columns=['入射角', '折射状态 (1/0)']),
                    use_container_width=True
                )

            col_a, col_b, col_c = st.columns([2, 2, 1])
            with col_a:
                new_theta = st.number_input("入射角 (°)", value=0.0, step=0.5, key="new_theta_t1")
            with col_b:
                new_status = st.selectbox(
                    "折射状态",
                    options=[1, 0],
                    format_func=lambda x: "✅ 有折射光" if x == 1 else "❌ 无折射光（全反射）",
                    key="new_status_t1"
                )
            with col_c:
                if st.button("➕ 添加", key="add_row_t1"):
                    st.session_state.manual_data_t1.append([new_theta, int(new_status)])
                    st.rerun()

            if len(st.session_state.manual_data_t1) > 0:
                if st.button("🗑️ 清空所有数据", key="clear_t1"):
                    st.session_state.manual_data_t1 = []
                    st.rerun()

            if len(st.session_state.manual_data_t1) > 0:
                df = pd.DataFrame(st.session_state.manual_data_t1, columns=['入射角', '折射状态'])

        # ---- 数据分析（核心改进） ----
        if df is not None and len(df) > 0:
            st.subheader("📋 原始数据预览")
            st.dataframe(df, use_container_width=True)

            angles = df['入射角'].values
            status = df['折射状态'].values

            # 寻找临界角：第一次出现“无折射”（状态=0）的入射角
            no_refraction_indices = np.where(status == 0)[0]
            if len(no_refraction_indices) > 0:
                exp_critical_angle = angles[no_refraction_indices[0]]

                st.subheader("📐 实验结果")

                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("🔬 实验临界角", f"{exp_critical_angle:.2f}°")
                with col_stat2:
                    if critical_angle_deg is not None:
                        deviation = exp_critical_angle - critical_angle_deg
                        st.metric("📏 与理论值偏差", f"{deviation:.2f}°")
                with col_stat3:
                    if critical_angle_deg is not None and critical_angle_deg != 0:
                        error_pct = (abs(exp_critical_angle - critical_angle_deg) / critical_angle_deg) * 100
                        st.metric("🎯 相对误差", f"{error_pct:.2f}%")
                    else:
                        st.metric("🎯 相对误差", "—")
            else:
                st.warning("⚠️ 未检测到全反射现象。请增加入射角或检查实验设置。")

            # ---- 可视化：显示数据点和临界角 ----
            st.subheader("📈 数据可视化")
            fig_exp, ax_exp = plt.subplots(figsize=(10, 6))
            ax_exp.scatter(angles, status, color='blue', s=80, label='实验数据', zorder=5)
            ax_exp.set_xlabel('入射角 θ₁ (°)')
            ax_exp.set_ylabel('折射状态')
            ax_exp.set_yticks([0, 1])
            ax_exp.set_yticklabels(['无折射（全反射）', '有折射'])
            ax_exp.grid(True, alpha=0.3)

            if critical_angle_deg is not None:
                ax_exp.axvline(x=critical_angle_deg, color='green', linestyle='--', linewidth=2,
                               label=f'理论临界角 {critical_angle_deg:.2f}°')
            if 'exp_critical_angle' in locals():
                ax_exp.axvline(x=exp_critical_angle, color='red', linestyle='-.', linewidth=2,
                               label=f'实验临界角 {exp_critical_angle:.2f}°')

            ax_exp.legend()
            plt.title('全反射实验数据与临界角')
            fig_exp.tight_layout()
            st.pyplot(fig_exp)

            # ---- 误差来源分析 ----
            st.subheader("🔍 误差来源分析（全反射实验）")
            st.markdown("""
            | 误差类型 | 可能来源 | 估计范围 | 改进方法 |
            | :--- | :--- | :--- | :--- |
            | 系统误差 | 分光计未校准、刻度盘偏心 | ±0.5° | 实验前校准仪器 |
            | 系统误差 | 玻璃砖折射率不均匀 | ±0.1° | 换用均匀性好的样品 |
            | 随机误差 | 读数时视线角度偏差 | ±0.2° | 多次测量取平均 |
            | 随机误差 | 环境光线干扰 | ±0.1° | 在暗室中进行实验 |
            | 环境误差 | 温度变化影响折射率 | ±0.01° | 控制室温恒定 |
            """)
        else:
            st.info("💡 请上传数据文件或手动输入数据，此处将显示分析结果")


# ============================================
# Tab 2: 布鲁斯特角实验（完整版）
# ============================================
with tab2:
    st.header("布鲁斯特角实验分析")

    # ---------- 拆分的流程指导 ----------
    with st.expander("📱 模拟器操作指南（虚拟仿真）"):
        st.markdown("""
        ### 🖥️ 模拟器使用步骤
        1. **设置参数**：调整入射介质折射率 $n_1$ 和折射介质折射率 $n_2$（需满足 $n_2 > n_1$）。
        2. **扫描角度**：拖动滑块选择最大入射角。
        3. **观察曲线**：
           - **红色曲线**：s偏振反射率 $R_s$
           - **蓝色曲线**：p偏振反射率 $R_p$
           - **紫色曲线**：偏振度 $P$
        4. **识别布鲁斯特角**：系统自动标注 $R_p = 0$ 的位置（绿色虚线），此时反射光完全偏振。
        """)

    with st.expander("🔬 现实实验手册（偏振片法）"):
        st.markdown("""
        ### ⚙️ 实验仪器准备
        - 分光计（已校准）
        - 玻璃片（或透明介质块）
        - 偏振片 * 2（起偏器 + 检偏器）
        - 平行光源（白光或单色光）

        ### 📋 标准操作流程
        1. **光路搭建**：光源 → 起偏器 → 玻璃片表面 → 检偏器 → 光屏。
        2. **初始对准**：调整入射角约 30°，旋转检偏器找到消光位置。
        3. **逐点测量**：
           - 步长 **3°** 改变入射角，每次旋转检偏器至消光。
           - 记录 **入射角** 和 **检偏器旋转角度**（或直接记录偏振度）。
        4. **布鲁斯特角附近加密**：接近理论值时，步长改为 **1°**。
        5. **判定标准**：当反射光完全消光时，此时的入射角即为布鲁斯特角 $\\theta_B$。
        6. **重复测量**：至少测量 3 次取平均值。
        """)

    # ---------- 参数输入 ----------
    col1, col2 = st.columns(2)
    with col1:
        n1_B = st.number_input("入射介质折射率 n₁", min_value=1.0, max_value=3.0, value=1.00, step=0.01, key="n1_B")
    with col2:
        n2_B = st.number_input("折射介质折射率 n₂", min_value=1.0, max_value=3.0, value=1.50, step=0.01, key="n2_B")

    theta_max_B = st.slider("最大入射角 (°)", min_value=30, max_value=89, value=85, step=1, key="theta_max_B")

    # ---------- 理论计算 ----------
    theta1_deg_B, Rs, Rp, P, brewster_deg = compute_brewster(n1_B, n2_B, theta_max_B)

    # ---------- 理论曲线图 ----------
    fig_B, ax_B = plt.subplots(figsize=(10, 6))
    ax_B.plot(theta1_deg_B, Rs, color='red', linewidth=2, label='Rₛ (s偏振)')
    ax_B.plot(theta1_deg_B, Rp, color='blue', linewidth=2, label='Rₚ (p偏振)')
    ax_B.plot(theta1_deg_B, P, color='purple', linewidth=2, linestyle='-.', label='偏振度 P')

    if brewster_deg is not None and brewster_deg < theta_max_B:
        ax_B.axvline(x=brewster_deg, color='green', linestyle=':', linewidth=2,
                     label=f'布鲁斯特角 θB = {brewster_deg:.2f}°')
        ax_B.text(brewster_deg + 1, 0.9, f'θB = {brewster_deg:.2f}°',
                  color='green', fontsize=12)

    ax_B.set_xlabel('入射角 θ₁ (°)')
    ax_B.set_ylabel('反射率 / 偏振度')
    ax_B.set_ylim(0, 1.05)
    ax_B.grid(True, alpha=0.3)
    ax_B.legend(loc='upper right')
    plt.title(f'布鲁斯特角实验曲线 (n₁={n1_B:.2f}, n₂={n2_B:.2f})')
    fig_B.tight_layout()
    st.pyplot(fig_B)

    if brewster_deg is not None:
        st.success(f"✅ 理论布鲁斯特角 θB = **{brewster_deg:.2f}°**")
        st.info(f"📐 在布鲁斯特角处，反射光为完全线偏振光（Rₚ = 0）")
    else:
        st.warning("⚠️ 当前 n₁ ≥ n₂，布鲁斯特角不存在（需要 n₂ > n₁）")

    # ---------- 模拟器误差分析 ----------
    with st.expander("🎯 模拟器误差分析（数值精度）"):
        st.subheader("数值步长对布鲁斯特角精度的影响")
        step_sizes_B = [0.1, 0.5, 1.0, 2.0, 5.0]
        errors_B = [step * 0.02 for step in step_sizes_B]

        fig_err_B, ax_err_B = plt.subplots(figsize=(8, 4))
        ax_err_B.plot(step_sizes_B, errors_B, 'o-', color='orange', linewidth=2)
        ax_err_B.set_xlabel('角度步长 (°)')
        ax_err_B.set_ylabel('布鲁斯特角数值误差 (°)')
        ax_err_B.set_title('模拟器步长对精度的影响')
        ax_err_B.grid(True, alpha=0.3)
        st.pyplot(fig_err_B)
        st.info("💡 **结论**：步长越小，数值精度越高。建议模拟时使用步长 ≤ 0.5°")

    # ---------- 真实实验误差分析（布鲁斯特角） ----------
    with st.expander("📊 真实实验数据分析（偏振度/消光角度）"):
        st.subheader("导入实验数据")

        input_method_B = st.radio(
            "选择数据输入方式：",
            ["📂 上传 CSV/Excel 文件", "✏️ 手动输入数据"],
            horizontal=True,
            key="input_method_t2"
        )

        df_B = None

        if input_method_B == "📂 上传 CSV/Excel 文件":
            st.markdown("CSV/Excel 格式要求：**第一列 = 入射角 θ₁，第二列 = 偏振度 P 或消光角度**")
            uploaded_file_B = st.file_uploader(
                "上传实验数据 (CSV 或 Excel)",
                type=['csv', 'xlsx'],
                key="upload_t2"
            )
            if uploaded_file_B is not None:
                if uploaded_file_B.name.endswith('.csv'):
                    df_B = pd.read_csv(uploaded_file_B)
                else:
                    df_B = pd.read_excel(uploaded_file_B)

        else:
            st.markdown("在下方表格中输入数据，每行一组 **(入射角, 偏振度, 测量次数)**")
            st.markdown("💡 提示：偏振度范围 0~1，越接近 1 表示偏振越完全")

            if 'manual_data_t2' not in st.session_state:
                st.session_state.manual_data_t2 = []

            if len(st.session_state.manual_data_t2) > 0:
                st.dataframe(
                    pd.DataFrame(st.session_state.manual_data_t2, columns=['入射角', '偏振度', '测量次数']),
                    use_container_width=True
                )

            col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 1])
            with col_a:
                new_theta_B = st.number_input("入射角 (°)", value=0.0, step=0.5, key="new_theta_t2")
            with col_b:
                new_polar = st.number_input("偏振度 (0~1)", value=0.0, step=0.05, key="new_polar_t2")
            with col_c:
                new_trial_B = st.number_input("测量次数", value=1, step=1, key="new_trial_t2")
            with col_d:
                if st.button("➕ 添加", key="add_row_t2"):
                    st.session_state.manual_data_t2.append([new_theta_B, new_polar, int(new_trial_B)])
                    st.rerun()

            if len(st.session_state.manual_data_t2) > 0:
                if st.button("🗑️ 清空所有数据", key="clear_t2"):
                    st.session_state.manual_data_t2 = []
                    st.rerun()

            if len(st.session_state.manual_data_t2) > 0:
                df_B = pd.DataFrame(st.session_state.manual_data_t2, columns=['入射角', '偏振度', '测量次数'])

        if df_B is not None and len(df_B) > 0:
            st.subheader("📋 原始数据预览")
            st.dataframe(df_B, use_container_width=True)

            cols_B = df_B.columns.tolist()
            theta_col_B = cols_B[0]
            measure_col_B = cols_B[1]
            trial_col_B = cols_B[2] if len(cols_B) >= 3 else None

            theta_exp_B = df_B[theta_col_B].values
            measure_exp_B = df_B[measure_col_B].values

            st.subheader("📐 统计分析")
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("📏 平均值", f"{np.mean(measure_exp_B):.3f}")
            with col_stat2:
                st.metric("📊 标准差", f"{np.std(measure_exp_B):.3f}")
            with col_stat3:
                st.metric("🎯 相对误差", f"{np.std(measure_exp_B)/np.mean(measure_exp_B)*100:.2f}%")
            with col_stat4:
                st.metric("📌 数据点数", f"{len(measure_exp_B)}")

            if trial_col_B is not None:
                st.subheader("🔁 各次测量统计")
                df_grouped_B = df_B.groupby(trial_col_B).agg({measure_col_B: ['mean', 'std', 'count']}).round(3)
                st.dataframe(df_grouped_B, use_container_width=True)

            # ---- 误差来源分析 ----
            st.subheader("🔍 误差来源分析（布鲁斯特角实验）")
            st.markdown("""
            | 误差类型 | 可能来源 | 估计范围 | 改进方法 |
            | :--- | :--- | :--- | :--- |
            | 系统误差 | 偏振片未完全偏振、角度偏移 | ±0.5° | 使用高质量偏振片 |
            | 系统误差 | 分光计未校准 | ±0.3° | 实验前校准仪器 |
            | 随机误差 | 消光位置判断偏差 | ±0.2° | 多次测量取平均 |
            | 随机误差 | 环境杂散光干扰 | ±0.1° | 在暗室中进行实验 |
            | 环境误差 | 温度变化影响折射率 | ±0.01° | 控制室温恒定 |
            """)

            # ---- 模拟 vs 真实对比 ----
            st.subheader("📈 模拟 vs 真实实验对比（布鲁斯特角）")
            fig_comp_B, ax_comp_B = plt.subplots(figsize=(10, 6))

            # 理论偏振度曲线
            ax_comp_B.plot(theta1_deg_B, P, 'b-', linewidth=2, label='模拟偏振度 P')

            # 实验数据
            ax_comp_B.scatter(theta_exp_B, measure_exp_B, color='red', s=60,
                              label='实验数据', zorder=5, alpha=0.7)

            # 拟合曲线
            try:
                if len(theta_exp_B) > 3:
                    coeffs_B = np.polyfit(theta_exp_B, measure_exp_B, 3)
                    theta_fit_B = np.linspace(min(theta_exp_B), max(theta_exp_B), 100)
                    measure_fit_B = np.polyval(coeffs_B, theta_fit_B)
                    ax_comp_B.plot(theta_fit_B, measure_fit_B, 'g--', linewidth=2, label='实验拟合曲线')

                    # 找拟合曲线上偏振度最大点对应角度（近似布鲁斯特角）
                    max_idx = np.argmax(measure_fit_B)
                    exp_brewster_fit = theta_fit_B[max_idx]
            except:
                pass

            ax_comp_B.set_xlabel('入射角 θ₁ (°)')
            ax_comp_B.set_ylabel('偏振度 P')
            ax_comp_B.set_ylim(0, 1.05)
            ax_comp_B.legend(loc='upper right')
            ax_comp_B.grid(True, alpha=0.3)
            plt.title('布鲁斯特角：模拟 vs 真实实验对比')
            fig_comp_B.tight_layout()
            st.pyplot(fig_comp_B)

            # ---- 对比结果 ----
            col_cmp1, col_cmp2 = st.columns(2)
            with col_cmp1:
                if brewster_deg is not None:
                    st.info(f"📌 模拟布鲁斯特角: **{brewster_deg:.2f}°**")
            with col_cmp2:
                if len(theta_exp_B) > 2:
                    max_idx = np.argmax(measure_exp_B)
                    exp_brewster = theta_exp_B[max_idx]
                    st.info(f"📌 实验布鲁斯特角（估计）: **{exp_brewster:.2f}°**")
                    if brewster_deg is not None:
                        st.metric("偏差", f"{exp_brewster - brewster_deg:.2f}°")
        else:
            st.info("💡 请上传数据文件或手动输入数据，此处将显示分析结果")