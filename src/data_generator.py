"""
Synthetic Salary Dataset Generator for 2025
Creates realistic salary data with modern features and complex correlations
"""

import numpy as np
import pandas as pd
from datetime import datetime
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

class SalaryDataGenerator:
    """Generate realistic synthetic salary dataset for 2025"""

    def __init__(self, n_samples=50000):
        self.n_samples = n_samples
        self.current_year = 2025

        # Define categories
        self.genders = ['Male', 'Female', 'Non-binary']
        self.races = ['White', 'Black', 'Asian', 'Hispanic', 'Other']
        self.education_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
        self.job_categories = ['Tech', 'Finance', 'Healthcare', 'Marketing', 'Sales',
                              'Operations', 'Data Science', 'Product', 'Engineering', 'Legal']
        self.work_modes = ['Remote', 'Hybrid', 'Onsite']
        self.company_sizes = ['Startup (1-50)', 'Small (51-200)', 'Medium (201-1000)',
                             'Large (1001-5000)', 'Enterprise (5000+)']
        self.countries = ['USA', 'UK', 'Canada', 'Germany', 'India', 'Singapore',
                         'Australia', 'France', 'Netherlands', 'Brazil']
        self.departments = ['Engineering', 'Product', 'Data', 'Marketing', 'Sales',
                           'Finance', 'HR', 'Operations', 'Legal', 'Executive']

        # Job titles by seniority and category
        self.job_titles = {
            'Tech': ['Junior Developer', 'Software Engineer', 'Senior Engineer',
                    'Lead Engineer', 'Principal Engineer', 'Engineering Manager',
                    'Director of Engineering', 'VP Engineering', 'CTO'],
            'Data Science': ['Junior Data Analyst', 'Data Analyst', 'Data Scientist',
                           'Senior Data Scientist', 'Lead Data Scientist',
                           'ML Engineer', 'Staff ML Engineer', 'Head of Data Science'],
            'Finance': ['Financial Analyst', 'Senior Financial Analyst', 'Finance Manager',
                       'Senior Finance Manager', 'Finance Director', 'VP Finance', 'CFO'],
            'Marketing': ['Marketing Coordinator', 'Marketing Manager', 'Senior Marketing Manager',
                         'Marketing Director', 'VP Marketing', 'CMO'],
            'Sales': ['Sales Rep', 'Account Executive', 'Senior Account Executive',
                     'Sales Manager', 'Sales Director', 'VP Sales'],
            'Product': ['Associate PM', 'Product Manager', 'Senior PM', 'Lead PM',
                       'Director of Product', 'VP Product', 'CPO'],
            'Engineering': ['Junior Engineer', 'Engineer', 'Senior Engineer', 'Staff Engineer',
                          'Principal Engineer', 'Distinguished Engineer'],
            'Healthcare': ['Nurse', 'Senior Nurse', 'Physician', 'Senior Physician',
                          'Department Head', 'Medical Director'],
            'Operations': ['Operations Analyst', 'Operations Manager', 'Senior Ops Manager',
                          'Director of Operations', 'VP Operations', 'COO'],
            'Legal': ['Paralegal', 'Associate Attorney', 'Senior Attorney', 'Legal Counsel',
                     'Senior Counsel', 'General Counsel']
        }

        # Economic indices by country (normalized to 100 for USA)
        self.economic_indices = {
            'USA': 100, 'UK': 92, 'Canada': 88, 'Germany': 95, 'India': 35,
            'Singapore': 110, 'Australia': 93, 'France': 89, 'Netherlands': 96, 'Brazil': 42
        }

        # Cost of living indices
        self.col_indices = {
            'USA': 100, 'UK': 85, 'Canada': 78, 'Germany': 75, 'India': 28,
            'Singapore': 95, 'Australia': 83, 'France': 81, 'Netherlands': 80, 'Brazil': 45
        }

    def generate_dataset(self):
        """Generate complete dataset with realistic correlations"""

        print("Generating synthetic salary dataset for 2025...")

        # Initialize data dictionary
        data = {}

        # Basic demographics
        data['age'] = np.random.normal(35, 8, self.n_samples).clip(22, 65).astype(int)
        data['gender'] = np.random.choice(self.genders, self.n_samples,
                                         p=[0.52, 0.46, 0.02])  # Realistic distribution
        data['race'] = np.random.choice(self.races, self.n_samples,
                                       p=[0.60, 0.13, 0.18, 0.06, 0.03])

        # Education (age-correlated)
        education_probs = self._get_education_probs(data['age'])
        data['education_level'] = [np.random.choice(self.education_levels, p=probs)
                                   for probs in education_probs]

        # Experience (age and education correlated)
        data['years_of_experience'] = self._calculate_experience(data['age'],
                                                                 data['education_level'])

        # Job category and title
        data['job_category'] = np.random.choice(self.job_categories, self.n_samples,
                                               p=[0.18, 0.12, 0.10, 0.10, 0.09,
                                                  0.08, 0.15, 0.08, 0.07, 0.03])

        data['job_title'] = [self._assign_job_title(cat, exp, edu)
                            for cat, exp, edu in zip(data['job_category'],
                                                     data['years_of_experience'],
                                                     data['education_level'])]

        # Company attributes
        data['company_size'] = np.random.choice(self.company_sizes, self.n_samples,
                                               p=[0.15, 0.25, 0.30, 0.20, 0.10])
        data['company_location'] = np.random.choice(self.countries, self.n_samples,
                                                    p=[0.40, 0.10, 0.08, 0.08, 0.12,
                                                       0.05, 0.06, 0.05, 0.04, 0.02])

        # Work mode (trend towards remote/hybrid in 2025)
        data['work_mode'] = np.random.choice(self.work_modes, self.n_samples,
                                            p=[0.35, 0.45, 0.20])

        # Department
        data['department'] = [self._map_category_to_department(cat)
                             for cat in data['job_category']]

        # Performance metrics
        data['performance_rating'] = np.random.beta(8, 2, self.n_samples) * 5  # Skewed towards high performance
        data['manager_rating'] = np.random.beta(7, 2, self.n_samples) * 5

        # Advanced features
        data['certifications_count'] = self._generate_certifications(data['job_category'],
                                                                     data['years_of_experience'])
        data['github_portfolio_strength'] = self._generate_github_score(data['job_category'])
        data['linkedin_connections'] = np.random.lognormal(5.5, 1.2, self.n_samples).clip(50, 30000).astype(int)

        # Skills and proficiency
        data['programming_languages_known'] = self._generate_programming_skills(data['job_category'])
        data['ai_ml_tools_proficiency'] = self._generate_ai_proficiency(data['job_category'],
                                                                        data['years_of_experience'])

        # University ranking (for degree holders)
        data['highest_degree_university_rank'] = self._generate_university_rank(data['education_level'])

        # Work intensity
        data['overtime_hours_per_month'] = np.random.gamma(2, 8, self.n_samples).clip(0, 80).astype(int)

        # Economic factors
        data['economic_index_of_country'] = [self.economic_indices[loc] for loc in data['company_location']]
        data['cost_of_living_index'] = [self.col_indices[loc] for loc in data['company_location']]

        # City tier (affects salary)
        data['city_tier'] = np.random.choice([1, 2, 3], self.n_samples, p=[0.40, 0.35, 0.25])

        # Negotiation ability
        data['salary_negotiation_score'] = np.random.beta(5, 5, self.n_samples) * 10

        # Previous salary (for experienced workers)
        data['previous_salary_usd'] = self._generate_previous_salary(data['years_of_experience'])

        # Calculate base salary with complex correlations
        base_salary = self._calculate_base_salary(data)

        # Add bonus and stock options
        data['bonus_percentage'] = self._calculate_bonus(data['job_category'],
                                                         data['performance_rating'],
                                                         data['company_size'])
        data['stock_options_value'] = self._calculate_stock_options(data['job_category'],
                                                                    data['company_size'],
                                                                    base_salary)

        # Final salary (with biases and noise)
        data['annual_salary_usd'] = self._apply_biases_and_noise(base_salary, data)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Add unique ID
        df.insert(0, 'employee_id', [f'EMP{str(i).zfill(6)}' for i in range(len(df))])

        print(f"Dataset generated successfully with {len(df)} rows and {len(df.columns)} columns")

        return df

    def _get_education_probs(self, ages):
        """Generate education probabilities based on age"""
        probs = []
        for age in ages:
            if age < 25:
                p = [0.30, 0.20, 0.45, 0.05, 0.00]
            elif age < 30:
                p = [0.15, 0.15, 0.50, 0.18, 0.02]
            elif age < 40:
                p = [0.10, 0.12, 0.45, 0.28, 0.05]
            elif age < 50:
                p = [0.12, 0.13, 0.40, 0.28, 0.07]
            else:
                p = [0.15, 0.15, 0.35, 0.25, 0.10]
            probs.append(p)
        return probs

    def _calculate_experience(self, ages, education_levels):
        """Calculate years of experience based on age and education"""
        experience = []
        edu_years = {'High School': 0, 'Associate': 2, 'Bachelor': 4, 'Master': 6, 'PhD': 10}

        for age, edu in zip(ages, education_levels):
            start_age = 18 + edu_years[edu]
            max_exp = max(0, age - start_age)
            # Add some randomness
            exp = max(0, int(np.random.normal(max_exp * 0.85, 3)))
            experience.append(min(exp, max_exp))

        return experience

    def _assign_job_title(self, category, experience, education):
        """Assign job title based on category, experience, and education"""
        if category not in self.job_titles:
            category = 'Tech'

        titles = self.job_titles[category]

        # Seniority index based on experience and education
        edu_boost = {'High School': 0, 'Associate': 0.5, 'Bachelor': 1, 'Master': 1.5, 'PhD': 2}
        seniority = min(experience + edu_boost[education], len(titles) - 1)

        # Add randomness
        idx = int(np.random.normal(seniority, 1.5))
        idx = max(0, min(idx, len(titles) - 1))

        return titles[idx]

    def _map_category_to_department(self, category):
        """Map job category to department"""
        mapping = {
            'Tech': 'Engineering',
            'Data Science': 'Data',
            'Finance': 'Finance',
            'Marketing': 'Marketing',
            'Sales': 'Sales',
            'Product': 'Product',
            'Engineering': 'Engineering',
            'Healthcare': 'Operations',
            'Operations': 'Operations',
            'Legal': 'Legal'
        }
        return mapping.get(category, 'Operations')

    def _generate_certifications(self, categories, experiences):
        """Generate certification counts"""
        certs = []
        for cat, exp in zip(categories, experiences):
            if cat in ['Tech', 'Data Science', 'Engineering']:
                base = exp * 0.3
            elif cat in ['Finance', 'Legal']:
                base = exp * 0.25
            else:
                base = exp * 0.15

            count = int(np.random.poisson(base))
            certs.append(min(count, 20))

        return certs

    def _generate_github_score(self, categories):
        """Generate GitHub portfolio strength (0-100)"""
        scores = []
        for cat in categories:
            if cat in ['Tech', 'Data Science', 'Engineering']:
                score = np.random.beta(3, 2) * 100
            elif cat in ['Product']:
                score = np.random.beta(2, 3) * 100
            else:
                score = np.random.beta(1, 5) * 100

            scores.append(round(score, 2))

        return scores

    def _generate_programming_skills(self, categories):
        """Generate number of programming languages known"""
        skills = []
        for cat in categories:
            if cat in ['Tech', 'Data Science', 'Engineering']:
                count = int(np.random.gamma(3, 1.5))
            elif cat in ['Product', 'Finance']:
                count = int(np.random.gamma(1, 1))
            else:
                count = int(np.random.gamma(0.5, 0.8))

            skills.append(min(count, 15))

        return skills

    def _generate_ai_proficiency(self, categories, experiences):
        """Generate AI/ML tools proficiency (0-10)"""
        proficiency = []
        for cat, exp in zip(categories, experiences):
            if cat in ['Data Science']:
                base = min(10, 4 + exp * 0.3)
                prof = np.random.normal(base, 1.5)
            elif cat in ['Tech', 'Engineering']:
                base = min(10, 2 + exp * 0.2)
                prof = np.random.normal(base, 2)
            else:
                prof = np.random.exponential(1.5)

            proficiency.append(round(max(0, min(10, prof)), 2))

        return proficiency

    def _generate_university_rank(self, education_levels):
        """Generate university ranking (1-500, or 0 for no degree)"""
        ranks = []
        for edu in education_levels:
            if edu in ['High School']:
                ranks.append(0)
            elif edu == 'Associate':
                rank = int(np.random.uniform(200, 500))
                ranks.append(rank)
            else:
                # Better education correlates with better universities
                if edu == 'PhD':
                    rank = int(np.random.gamma(3, 30))
                elif edu == 'Master':
                    rank = int(np.random.gamma(5, 35))
                else:  # Bachelor
                    rank = int(np.random.gamma(7, 40))

                ranks.append(min(max(1, rank), 500))

        return ranks

    def _generate_previous_salary(self, experiences):
        """Generate previous salary for experienced workers"""
        salaries = []
        for exp in experiences:
            if exp == 0:
                salaries.append(0)
            else:
                # Previous salary increases with experience
                base = 40000 + exp * 4000
                salary = np.random.normal(base, base * 0.2)
                salaries.append(max(0, int(salary)))

        return salaries

    def _calculate_base_salary(self, data):
        """Calculate base salary with complex correlations"""

        # Start with base (2025 adjusted)
        salary = np.ones(self.n_samples) * 42000

        # Education multiplier
        edu_mult = {'High School': 1.0, 'Associate': 1.15, 'Bachelor': 1.40,
                    'Master': 1.75, 'PhD': 2.10}
        salary *= [edu_mult[edu] for edu in data['education_level']]

        # Experience (exponential growth early career, linear later)
        exp_bonus = [min(exp * 0.10, 4) for exp in data['years_of_experience']]
        salary *= (1 + np.array(exp_bonus))

        # Job category multiplier
        category_mult = {
            'Data Science': 1.25, 'Tech': 1.22, 'Engineering': 1.20,
            'Finance': 1.18, 'Product': 1.15, 'Legal': 1.14,
            'Sales': 1.05, 'Marketing': 1.03, 'Operations': 1.00, 'Healthcare': 1.08
        }
        salary *= [category_mult[cat] for cat in data['job_category']]

        # Seniority from job title
        seniority_bonus = [self._extract_seniority_multiplier(title)
                          for title in data['job_title']]
        salary *= seniority_bonus

        # Company size
        size_mult = {'Startup (1-50)': 0.90, 'Small (51-200)': 0.95,
                    'Medium (201-1000)': 1.05, 'Large (1001-5000)': 1.15,
                    'Enterprise (5000+)': 1.25}
        salary *= [size_mult[size] for size in data['company_size']]

        # Location (economic index and city tier)
        salary *= np.array(data['economic_index_of_country']) / 100
        city_mult = {1: 1.20, 2: 1.05, 3: 0.92}
        salary *= [city_mult[tier] for tier in data['city_tier']]

        # Work mode (remote commands premium in 2025)
        mode_mult = {'Remote': 1.08, 'Hybrid': 1.03, 'Onsite': 1.00}
        salary *= [mode_mult[mode] for mode in data['work_mode']]

        # Performance
        salary *= (0.85 + data['performance_rating'] * 0.06)

        # Skills premium
        salary *= (1 + np.array(data['programming_languages_known']) * 0.02)
        salary *= (1 + np.array(data['ai_ml_tools_proficiency']) * 0.015)

        # GitHub portfolio (for tech roles)
        tech_roles = np.array([1 if cat in ['Tech', 'Data Science', 'Engineering']
                              else 0 for cat in data['job_category']])
        salary *= (1 + tech_roles * np.array(data['github_portfolio_strength']) * 0.001)

        # University ranking (inverse relationship - lower rank number = better)
        uni_bonus = [0.05 if rank > 0 and rank <= 50 else
                    0.03 if rank <= 100 else
                    0.01 if rank <= 200 else 0
                    for rank in data['highest_degree_university_rank']]
        salary *= (1 + np.array(uni_bonus))

        # Certifications
        salary *= (1 + np.array(data['certifications_count']) * 0.012)

        # Negotiation skill
        salary *= (1 + np.array(data['salary_negotiation_score']) * 0.008)

        # Previous salary influence (sticky wages)
        prev_sal_boost = [0.05 if prev > 0 else 0 for prev in data['previous_salary_usd']]
        salary *= (1 + np.array(prev_sal_boost))

        return salary

    def _extract_seniority_multiplier(self, title):
        """Extract seniority multiplier from job title"""
        title_lower = title.lower()

        if any(word in title_lower for word in ['cto', 'cfo', 'cmo', 'cpo', 'coo', 'ceo']):
            return 2.80
        elif any(word in title_lower for word in ['vp', 'vice president']):
            return 2.30
        elif any(word in title_lower for word in ['director', 'head of']):
            return 1.85
        elif any(word in title_lower for word in ['principal', 'distinguished', 'staff']):
            return 1.60
        elif any(word in title_lower for word in ['lead', 'manager']):
            return 1.40
        elif any(word in title_lower for word in ['senior', 'sr']):
            return 1.25
        elif any(word in title_lower for word in ['junior', 'associate', 'jr']):
            return 0.85
        else:
            return 1.05

    def _calculate_bonus(self, categories, performance, company_sizes):
        """Calculate bonus percentage"""
        bonuses = []

        for cat, perf, size in zip(categories, performance, company_sizes):
            # Base bonus by category
            if cat in ['Finance', 'Sales']:
                base = 18
            elif cat in ['Tech', 'Data Science', 'Product']:
                base = 12
            else:
                base = 8

            # Performance multiplier
            perf_mult = 0.5 + (perf / 5) * 0.8

            # Company size
            if 'Enterprise' in size or 'Large' in size:
                size_mult = 1.3
            else:
                size_mult = 1.0

            bonus = base * perf_mult * size_mult
            bonus = max(0, np.random.normal(bonus, bonus * 0.3))
            bonuses.append(round(bonus, 2))

        return bonuses

    def _calculate_stock_options(self, categories, company_sizes, base_salaries):
        """Calculate stock options value"""
        stock_values = []

        for cat, size, salary in zip(categories, company_sizes, base_salaries):
            # Startups and tech companies offer more stock
            if 'Startup' in size:
                if cat in ['Tech', 'Data Science', 'Product', 'Engineering']:
                    stock_mult = np.random.uniform(0.15, 0.40)
                else:
                    stock_mult = np.random.uniform(0.05, 0.20)
            elif cat in ['Tech', 'Data Science', 'Engineering'] and 'Enterprise' in size:
                stock_mult = np.random.uniform(0.08, 0.25)
            else:
                stock_mult = np.random.uniform(0, 0.10)

            stock = salary * stock_mult
            stock_values.append(int(stock))

        return stock_values

    def _apply_biases_and_noise(self, base_salary, data):
        """Apply realistic biases and random noise to salary"""

        salary = base_salary.copy()

        # Gender pay gap (decreasing in 2025 but still present)
        gender_mult = {'Male': 1.00, 'Female': 0.94, 'Non-binary': 0.93}
        salary *= [gender_mult[g] for g in data['gender']]

        # Racial bias (subtle but measurable)
        race_mult = {'White': 1.00, 'Asian': 0.98, 'Black': 0.92,
                    'Hispanic': 0.93, 'Other': 0.95}
        salary *= [race_mult[r] for r in data['race']]

        # Overtime correlation (slight penalty for work-life balance in modern companies)
        overtime_penalty = 1 - (np.array(data['overtime_hours_per_month']) / 1000)
        salary *= overtime_penalty

        # LinkedIn network effect
        network_boost = np.log1p(np.array(data['linkedin_connections'])) / 50
        salary *= (1 + network_boost)

        # Add realistic noise
        noise = np.random.normal(1.0, 0.08, self.n_samples)
        salary *= noise

        # Ensure minimum wage and realistic caps
        salary = np.clip(salary, 35000, 550000)

        # Round to nearest 1000
        salary = (salary / 1000).round() * 1000

        return salary.astype(int)


def main():
    """Generate and save the dataset"""

    generator = SalaryDataGenerator(n_samples=50000)
    df = generator.generate_dataset()

    # Save dataset
    output_path = '../data/salary_data_2025.csv'
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")

    # Print basic statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"\nSalary Statistics:")
    print(df['annual_salary_usd'].describe())
    print(f"\nGender Distribution:")
    print(df['gender'].value_counts())
    print(f"\nJob Category Distribution:")
    print(df['job_category'].value_counts())
    print(f"\nEducation Distribution:")
    print(df['education_level'].value_counts())

    # Check for missing values
    print(f"\nMissing Values: {df.isnull().sum().sum()}")

    return df


if __name__ == "__main__":
    df = main()
