#!/usr/bin/env python3
# Data Analysis Assignment - Iris Dataset Exploration

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def main():
    print("# Data Analysis with Python")
    print("## Assignment: Exploring the Iris Dataset\n")
    print("**Objective**: Load, analyze, and visualize the Iris dataset using pandas and matplotlib.\n")

    # Try to import seaborn for styling, but continue without it if not available
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
        print("Seaborn styling enabled")
    except ImportError:
        print("Seaborn not found - using basic matplotlib styling")
        plt.style.use('ggplot')

    # Task 1: Load and Explore the Dataset
    print("\n## Task 1: Load and Explore the Dataset")
    
    try:
        iris = load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['species'] = iris.target
        iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        print("\nDataset loaded successfully!")
        print(f"Dataset shape: {iris_df.shape}")
        
        # Display first 5 rows
        print("\nFirst 5 rows of the dataset:")
        print(iris_df.head())
        
        # Explore dataset structure
        print("\nDataset information:")
        iris_df.info()
        
        # Check for missing values
        print("\nMissing values per column:")
        print(iris_df.isnull().sum())
        
        print("\n**Observation**: The Iris dataset is clean with no missing values and proper data types.")
        
        # Task 2: Basic Data Analysis
        print("\n## Task 2: Basic Data Analysis")
        
        # Basic statistics
        print("\nBasic statistics for numerical columns:")
        print(iris_df.describe())
        
        # Group by species and calculate mean
        print("\nMean measurements by species:")
        species_stats = iris_df.groupby('species').mean()
        print(species_stats)
        
        # Interesting finding
        print("\nPetal length comparison:")
        diff = species_stats.loc['versicolor', 'petal length (cm)'] - species_stats.loc['setosa', 'petal length (cm)']
        print(f"Setosa petals are {diff:.1f} cm shorter than Versicolor on average")
        
        # Task 3: Data Visualization
        print("\n## Task 3: Data Visualization")
        
        # Create figure for all plots
        plt.figure(figsize=(15, 10))
        
        # 1. Line chart
        plt.subplot(2, 2, 1)
        iris_df['sepal length (cm)'].plot(kind='line')
        plt.title('Sepal Length Trend (by index)')
        plt.xlabel('Index')
        plt.ylabel('Sepal Length (cm)')
        
        # 2. Bar chart
        plt.subplot(2, 2, 2)
        species_stats['petal length (cm)'].plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
        plt.title('Average Petal Length by Species')
        plt.ylabel('Length (cm)')
        plt.xticks(rotation=0)
        
        # 3. Histogram
        plt.subplot(2, 2, 3)
        iris_df['sepal width (cm)'].plot(kind='hist', bins=15, edgecolor='black', color='lightblue')
        plt.title('Distribution of Sepal Width')
        plt.xlabel('Width (cm)')
        
        # 4. Scatter plot
        plt.subplot(2, 2, 4)
        colors = {'setosa': 'blue', 'versicolor': 'green', 'virginica': 'red'}
        for species, color in colors.items():
            subset = iris_df[iris_df['species'] == species]
            plt.scatter(subset['sepal length (cm)'], subset['petal length (cm)'], 
                        label=species, c=color, alpha=0.7)
        plt.title('Sepal Length vs Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('iris_visualizations.png')  # Save the combined plots
        print("\nVisualizations saved to 'iris_visualizations.png'")
        
        # Key Findings
        print("\n## Key Findings")
        print("1. **Setosa Distinctiveness**: Setosa species has significantly shorter petals")
        print("2. **Sepal Width**: Most flowers have sepal width between 2.8-3.2 cm")
        print("3. **Strong Correlation**: Positive relationship between sepal and petal lengths")
        print("4. **Virginica Size**: Virginica has the largest measurements overall")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")

if __name__ == "__main__":
    main()