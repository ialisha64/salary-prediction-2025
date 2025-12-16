"""
Streamlit Salary Prediction App
Interactive web app for salary prediction with SHAP explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import sys
import os

# Add src to path
sys.path.append('src')

from predict import SalaryPredictor

# Page config
st.set_page_config(
    page_title="Salary Predictor 2025",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .status-overpaid {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .status-underpaid {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .status-fair {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """Load model and preprocessor (cached)"""
    try:
        predictor = SalaryPredictor(
            model_path='model/Voting_Ensemble.pkl',
            preprocessor_path='model/preprocessor.pkl'
        )
        return predictor
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first by running: python src/train.py")
        st.stop()


@st.cache_data
def load_dataset():
    """Load the dataset for dashboard"""
    try:
        df = pd.read_csv('data/salary_data_2025.csv')
        return df
    except FileNotFoundError:
        st.warning("Dataset not found. Some features may be unavailable.")
        return None


def create_input_form():
    """Create user input form"""

    st.sidebar.header("📝 Your Profile")

    with st.sidebar.expander("👤 Personal Information", expanded=True):
        age = st.slider("Age", 22, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
        race = st.selectbox("Race", ["White", "Black", "Asian", "Hispanic", "Other"])

    with st.sidebar.expander("🎓 Education & Experience", expanded=True):
        education_level = st.selectbox(
            "Education Level",
            ["High School", "Associate", "Bachelor", "Master", "PhD"]
        )
        years_of_experience = st.slider("Years of Experience", 0, 40, 5)
        university_rank = st.number_input(
            "University Ranking (1-500, 0 if N/A)",
            0, 500, 100
        )

    with st.sidebar.expander("💼 Job Details", expanded=True):
        job_category = st.selectbox(
            "Job Category",
            ["Tech", "Data Science", "Finance", "Marketing", "Sales",
             "Operations", "Product", "Engineering", "Healthcare", "Legal"]
        )

        # Job titles based on category
        job_titles_map = {
            'Tech': ['Junior Developer', 'Software Engineer', 'Senior Engineer',
                    'Lead Engineer', 'Engineering Manager', 'Director of Engineering'],
            'Data Science': ['Junior Data Analyst', 'Data Analyst', 'Data Scientist',
                           'Senior Data Scientist', 'ML Engineer', 'Head of Data Science'],
            'Finance': ['Financial Analyst', 'Senior Financial Analyst', 'Finance Manager',
                       'Finance Director', 'VP Finance'],
            'Marketing': ['Marketing Coordinator', 'Marketing Manager',
                         'Marketing Director', 'VP Marketing'],
            'Sales': ['Sales Rep', 'Account Executive', 'Sales Manager', 'Sales Director'],
            'Product': ['Associate PM', 'Product Manager', 'Senior PM', 'Director of Product'],
            'Engineering': ['Junior Engineer', 'Engineer', 'Senior Engineer', 'Staff Engineer'],
            'Healthcare': ['Nurse', 'Senior Nurse', 'Physician', 'Medical Director'],
            'Operations': ['Operations Analyst', 'Operations Manager', 'Director of Operations'],
            'Legal': ['Paralegal', 'Associate Attorney', 'Senior Attorney', 'Legal Counsel']
        }

        job_title = st.selectbox("Job Title", job_titles_map.get(job_category, ['Engineer']))

        department = st.selectbox(
            "Department",
            ["Engineering", "Data", "Finance", "Marketing", "Sales",
             "Product", "HR", "Operations", "Legal", "Executive"]
        )

    with st.sidebar.expander("🏢 Company Information", expanded=True):
        company_size = st.selectbox(
            "Company Size",
            ["Startup (1-50)", "Small (51-200)", "Medium (201-1000)",
             "Large (1001-5000)", "Enterprise (5000+)"]
        )
        company_location = st.selectbox(
            "Company Location",
            ["USA", "UK", "Canada", "Germany", "India", "Singapore",
             "Australia", "France", "Netherlands", "Brazil"]
        )
        city_tier = st.select_slider("City Tier", [1, 2, 3], value=1)
        work_mode = st.selectbox("Work Mode", ["Remote", "Hybrid", "Onsite"])

    with st.sidebar.expander("⭐ Performance & Skills", expanded=True):
        performance_rating = st.slider("Performance Rating (1-5)", 1.0, 5.0, 4.0, 0.1)
        manager_rating = st.slider("Manager Rating (1-5)", 1.0, 5.0, 4.0, 0.1)
        certifications_count = st.number_input("Number of Certifications", 0, 20, 2)
        programming_languages_known = st.number_input("Programming Languages Known", 0, 15, 3)
        ai_ml_proficiency = st.slider("AI/ML Tools Proficiency (0-10)", 0.0, 10.0, 5.0, 0.5)

    with st.sidebar.expander("🌐 Digital Presence", expanded=True):
        github_score = st.slider("GitHub Portfolio Strength (0-100)", 0.0, 100.0, 50.0, 1.0)
        linkedin_connections = st.number_input("LinkedIn Connections", 50, 30000, 500)

    with st.sidebar.expander("💪 Work & Negotiation", expanded=True):
        overtime_hours = st.slider("Overtime Hours/Month", 0, 80, 10)
        negotiation_score = st.slider("Negotiation Score (0-10)", 0.0, 10.0, 6.0, 0.5)
        previous_salary = st.number_input("Previous Salary (USD, 0 if first job)", 0, 500000, 0)

    # Economic indices (auto-filled based on location)
    economic_indices = {
        'USA': 100, 'UK': 92, 'Canada': 88, 'Germany': 95, 'India': 35,
        'Singapore': 110, 'Australia': 93, 'France': 89, 'Netherlands': 96, 'Brazil': 42
    }
    col_indices = {
        'USA': 100, 'UK': 85, 'Canada': 78, 'Germany': 75, 'India': 28,
        'Singapore': 95, 'Australia': 83, 'France': 81, 'Netherlands': 80, 'Brazil': 45
    }

    # Create input dictionary
    user_input = {
        'age': age,
        'gender': gender,
        'race': race,
        'education_level': education_level,
        'years_of_experience': years_of_experience,
        'job_title': job_title,
        'job_category': job_category,
        'company_size': company_size,
        'company_location': company_location,
        'work_mode': work_mode,
        'performance_rating': performance_rating,
        'manager_rating': manager_rating,
        'certifications_count': certifications_count,
        'github_portfolio_strength': github_score,
        'linkedin_connections': linkedin_connections,
        'programming_languages_known': programming_languages_known,
        'ai_ml_tools_proficiency': ai_ml_proficiency,
        'highest_degree_university_rank': university_rank,
        'overtime_hours_per_month': overtime_hours,
        'economic_index_of_country': economic_indices[company_location],
        'cost_of_living_index': col_indices[company_location],
        'city_tier': city_tier,
        'salary_negotiation_score': negotiation_score,
        'previous_salary_usd': previous_salary,
        'bonus_percentage': 10.0,  # Default
        'stock_options_value': 0,  # Default
        'department': department,
        'annual_salary_usd': 0
    }

    return user_input


def show_prediction_tab(predictor):
    """Show prediction interface"""

    st.markdown('<div class="main-header">💰 Salary Predictor 2025</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Welcome to the Advanced Salary Prediction System

    This AI-powered tool predicts your market salary based on 25+ factors including
    education, experience, skills, location, and performance metrics.

    **📌 How to use:**
    1. Fill in your profile in the sidebar
    2. Click 'Predict My Salary' button
    3. Get your predicted salary with confidence interval
    4. View personalized insights and career advice
    """)

    # Get user input
    user_input = create_input_form()

    # Prediction button
    if st.sidebar.button("🚀 Predict My Salary", type="primary", use_container_width=True):

        with st.spinner("Analyzing your profile..."):
            # Make prediction
            result = predictor.predict(user_input, return_confidence=True)

        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            ${result['predicted_salary']:,.0f}
        </div>
        """, unsafe_allow_html=True)

        # Confidence interval
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lower Bound", f"${result['lower_bound']:,.0f}")
        with col2:
            st.metric("Predicted Salary", f"${result['predicted_salary']:,.0f}")
        with col3:
            st.metric("Upper Bound", f"${result['upper_bound']:,.0f}")

        st.info(f"📊 Confidence Interval: ±{result['confidence_margin_pct']}% based on model performance")

        # Comparison section
        st.markdown("---")
        st.subheader("💼 Compare to Your Current Salary")

        current_salary = st.number_input(
            "Enter your current annual salary (USD)",
            min_value=0,
            max_value=1000000,
            value=0,
            step=1000
        )

        if current_salary > 0:
            comparison = predictor.compare_to_peers(user_input, current_salary)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Your Current Salary",
                    f"${comparison['actual_salary']:,.0f}"
                )

            with col2:
                st.metric(
                    "Difference from Market",
                    f"${comparison['difference']:,.0f}",
                    f"{comparison['pct_difference']:+.1f}%"
                )

            # Status message
            if comparison['status'] == 'OVERPAID':
                st.markdown(
                    f'<p class="status-overpaid">✅ You are earning {abs(comparison["pct_difference"]):.1f}% '
                    f'above the market average - Great negotiation!</p>',
                    unsafe_allow_html=True
                )
            elif comparison['status'] == 'UNDERPAID':
                st.markdown(
                    f'<p class="status-underpaid">⚠️ You are earning {abs(comparison["pct_difference"]):.1f}% '
                    f'below the market average - Consider negotiating for a raise!</p>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<p class="status-fair">✓ Your salary is within the fair market range</p>',
                    unsafe_allow_html=True
                )

        # Career advice
        st.markdown("---")
        st.subheader("🎯 Personalized Career Insights")

        insights = []

        # Experience-based insights
        if user_input['years_of_experience'] < 3:
            insights.append("🔹 **Early Career**: Focus on skill building and certifications to increase your value")
        elif user_input['years_of_experience'] < 7:
            insights.append("🔹 **Mid Career**: Consider leadership roles or specialization to boost earnings")
        else:
            insights.append("🔹 **Senior Professional**: Your experience is valuable - negotiate strongly!")

        # Education insights
        if user_input['education_level'] in ['High School', 'Associate']:
            insights.append("🔹 **Education**: A Bachelor's or Master's degree could increase your salary by 40-75%")

        # Skills insights
        if user_input['programming_languages_known'] < 3 and user_input['job_category'] in ['Tech', 'Data Science']:
            insights.append("🔹 **Skills**: Learning more programming languages could boost your salary by 10-15%")

        # Work mode insights
        if user_input['work_mode'] == 'Onsite':
            insights.append("🔹 **Work Mode**: Remote roles in your field typically pay 8% more - consider negotiating!")

        # Performance insights
        if user_input['performance_rating'] >= 4.5:
            insights.append("🔹 **Performance**: Your excellent ratings position you well for a raise or promotion")

        # GitHub insights
        if user_input['github_portfolio_strength'] < 50 and user_input['job_category'] in ['Tech', 'Data Science']:
            insights.append("🔹 **Portfolio**: Building your GitHub presence could increase offers by 5-10%")

        for insight in insights:
            st.markdown(insight)

        # Feature importance
        st.markdown("---")
        st.subheader("📊 Top Factors Affecting Your Salary")

        explanation = predictor.explain_prediction(user_input)
        if explanation is not None:
            top_5 = explanation.head(5)

            # Create horizontal bar chart
            fig = go.Figure(go.Bar(
                x=top_5['importance'],
                y=top_5['feature'],
                orientation='h',
                marker=dict(color='#667eea')
            ))

            fig.update_layout(
                title="Most Important Features for Your Prediction",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 View All Feature Importances"):
                st.dataframe(explanation, use_container_width=True)


def show_negotiation_simulator(predictor):
    """Salary negotiation simulator"""

    st.markdown('<div class="main-header">🎯 Salary Negotiation Simulator</div>', unsafe_allow_html=True)

    st.markdown("""
    ### What-If Analysis: See How Changes Impact Your Salary

    Adjust different factors to see how they would impact your predicted salary.
    This helps you understand what skills or changes would be most valuable.
    """)

    # Get base input
    user_input = create_input_form()

    st.markdown("---")
    st.subheader("🔄 Simulation Controls")

    col1, col2 = st.columns(2)

    scenarios = {}

    with col1:
        st.markdown("#### 📈 Improve Negotiation Skills")
        new_negotiation = st.slider(
            "What if negotiation score improved to:",
            0.0, 10.0,
            min(user_input['salary_negotiation_score'] + 2, 10.0),
            0.5
        )

        scenario_input = user_input.copy()
        scenario_input['salary_negotiation_score'] = new_negotiation
        scenarios['Better Negotiation'] = predictor.predict(scenario_input, return_confidence=False)

    with col2:
        st.markdown("#### 🎓 Advance Education")
        edu_levels = ["High School", "Associate", "Bachelor", "Master", "PhD"]
        current_idx = edu_levels.index(user_input['education_level'])
        if current_idx < len(edu_levels) - 1:
            new_education = edu_levels[current_idx + 1]

            scenario_input = user_input.copy()
            scenario_input['education_level'] = new_education
            scenarios[f'With {new_education}'] = predictor.predict(scenario_input, return_confidence=False)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 💻 Boost Technical Skills")
        new_prog_lang = st.number_input(
            "What if you knew more languages:",
            user_input['programming_languages_known'],
            15,
            min(user_input['programming_languages_known'] + 3, 15)
        )

        scenario_input = user_input.copy()
        scenario_input['programming_languages_known'] = new_prog_lang
        scenarios['More Programming Skills'] = predictor.predict(scenario_input, return_confidence=False)

    with col4:
        st.markdown("#### 🏠 Switch to Remote")
        if user_input['work_mode'] != 'Remote':
            scenario_input = user_input.copy()
            scenario_input['work_mode'] = 'Remote'
            scenarios['Remote Work'] = predictor.predict(scenario_input, return_confidence=False)

    # Show results
    if scenarios:
        st.markdown("---")
        st.subheader("📊 Scenario Comparison")

        base_salary = predictor.predict(user_input, return_confidence=False)

        comparison_data = {
            'Scenario': ['Current (Baseline)'] + list(scenarios.keys()),
            'Predicted Salary': [base_salary] + list(scenarios.values()),
        }

        df_scenarios = pd.DataFrame(comparison_data)
        df_scenarios['Increase'] = df_scenarios['Predicted Salary'] - base_salary
        df_scenarios['Increase %'] = (df_scenarios['Increase'] / base_salary * 100).round(2)

        # Create visualization
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_scenarios['Scenario'],
            y=df_scenarios['Predicted Salary'],
            text=[f"${val:,.0f}" for val in df_scenarios['Predicted Salary']],
            textposition='outside',
            marker=dict(
                color=df_scenarios['Predicted Salary'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Salary")
            )
        ))

        fig.update_layout(
            title="Salary Comparison Across Scenarios",
            xaxis_title="Scenario",
            yaxis_title="Predicted Salary (USD)",
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show table
        st.dataframe(
            df_scenarios.style.format({
                'Predicted Salary': '${:,.0f}',
                'Increase': '${:,.0f}',
                'Increase %': '{:+.2f}%'
            }),
            use_container_width=True
        )


def show_dashboard():
    """Show salary transparency dashboard"""

    st.markdown('<div class="main-header">📊 Salary Transparency Dashboard</div>', unsafe_allow_html=True)

    df = load_dataset()

    if df is None:
        st.error("Dataset not available for dashboard")
        return

    st.markdown("### Global Salary Trends & Insights")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Average Salary", f"${df['annual_salary_usd'].mean():,.0f}")

    with col2:
        st.metric("Median Salary", f"${df['annual_salary_usd'].median():,.0f}")

    with col3:
        st.metric("Max Salary", f"${df['annual_salary_usd'].max():,.0f}")

    with col4:
        st.metric("Total Employees", f"{len(df):,}")

    # Salary distribution
    st.markdown("---")
    st.subheader("💰 Salary Distribution")

    fig = px.histogram(
        df,
        x='annual_salary_usd',
        nbins=50,
        title="Overall Salary Distribution",
        labels={'annual_salary_usd': 'Annual Salary (USD)'},
        color_discrete_sequence=['#667eea']
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # By category
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📁 Salary by Job Category")
        avg_by_category = df.groupby('job_category')['annual_salary_usd'].mean().sort_values(ascending=True)

        fig = go.Figure(go.Bar(
            x=avg_by_category.values,
            y=avg_by_category.index,
            orientation='h',
            marker=dict(color='#764ba2')
        ))
        fig.update_layout(
            xaxis_title="Average Salary (USD)",
            yaxis_title="Job Category",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎓 Salary by Education")
        avg_by_education = df.groupby('education_level')['annual_salary_usd'].mean().sort_values(ascending=True)

        fig = go.Figure(go.Bar(
            x=avg_by_education.values,
            y=avg_by_education.index,
            orientation='h',
            marker=dict(color='#f093fb')
        ))
        fig.update_layout(
            xaxis_title="Average Salary (USD)",
            yaxis_title="Education Level",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Gender pay gap
    st.markdown("---")
    st.subheader("⚖️ Gender Pay Analysis")

    gender_stats = df.groupby('gender')['annual_salary_usd'].agg(['mean', 'median', 'count'])

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Mean', x=gender_stats.index, y=gender_stats['mean']))
    fig.add_trace(go.Bar(name='Median', x=gender_stats.index, y=gender_stats['median']))

    fig.update_layout(
        title="Average Salary by Gender",
        xaxis_title="Gender",
        yaxis_title="Salary (USD)",
        barmode='group',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Work mode analysis
    st.markdown("---")
    st.subheader("🏠 Remote vs Hybrid vs Onsite")

    work_mode_stats = df.groupby('work_mode')['annual_salary_usd'].mean().sort_values(ascending=False)

    fig = px.pie(
        values=work_mode_stats.values,
        names=work_mode_stats.index,
        title="Average Salary by Work Mode",
        color_discrete_sequence=px.colors.sequential.Purples
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main app"""

    # Load model
    predictor = load_model_and_preprocessor()

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    tab = st.sidebar.radio(
        "Choose a section:",
        ["💰 Salary Prediction", "🎯 Negotiation Simulator", "📊 Salary Dashboard"]
    )

    if tab == "💰 Salary Prediction":
        show_prediction_tab(predictor)
    elif tab == "🎯 Negotiation Simulator":
        show_negotiation_simulator(predictor)
    elif tab == "📊 Salary Dashboard":
        show_dashboard()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    **Salary Predictor 2025**

    Advanced ML-powered salary prediction using:
    - XGBoost / LightGBM / CatBoost
    - 25+ features
    - 50,000 data points
    - Real-time predictions

    Built with ❤️ using Streamlit
    """)


if __name__ == "__main__":
    main()
