import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)


def create_path():
    """
    @return: Paths for csv data

    Creates and returns pathnames for data
    """
    path = ''  # "/content/drive/My Drive/MA 346 Project 1/"
    file1 = 'https://raw.githubusercontent.com/sandhya0218/HappinessIndicators/main/world-happiness-report.csv'  # "world-happiness-report.csv"
    file2 = 'https://raw.githubusercontent.com/sandhya0218/HappinessIndicators/main/world-happiness-report-2021.csv'  # 'world-happiness-report-2021.csv'
    data1 = path + file1  # creating the paths to the files for the two sets of data
    data2 = path + file2
    return data1, data2


def domain_region(row, ref):
    """
    @param row: Array-type containing Country Name
    @param ref: DataFrame containing countries and their corresponding region
    @return: Region value

    Returns the corresponding region for a given country name
    """
    return ref.loc[row, 'region']


def clean_df(d1, d2):
    """
    @param d1: csv pathname of general Happiness Indicator data from 2005 to 2020
    @param d2: csv pathname of specific Happiness Indicator data from 2021
    @return: DataFrame of cleaned data

    Reads, merges, and cleans DataFrame/s
    """
    # Read both files
    df_partial = pd.read_csv(d1)
    df_2021 = pd.read_csv(d2)

    # Subset Country and corresponding Region from 2021 data
    df_2021 = df_2021[['Country name', 'Regional indicator']]

    # Summary of Data/Sanity Check
    # print(df_partial.isna().sum())
    # print(df_partial.describe().T)

    # Merge dataframes into one dataframe that includes region and preserves observations in df_partial
    df_full = df_partial.merge(df_2021, left_on='Country name', right_on='Country name', how='left')

    # Get medians for all numerical columns by country
    df_group = df_full.groupby('Country name')[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption', 'Positive affect', 'Negative affect']].agg('median')
    df_group.reset_index(inplace=True)

    # Merge dataframes, which results in duplicate columns: "_x" cols have data actual vals, "_y" cols have median vals by country
    df_comb = df_full.merge(df_group, left_on='Country name', right_on='Country name', how='left')

    # Fill null vals in "_x" cols with corresponding "_y" median value
    for col in df_full.columns[3:-1]:  # Slicing excludes columns that had no null vals or are not numeric
        df_comb[col + '_x'].fillna(df_comb[col+'_y'], inplace=True)

    # Extract numeric columns
    numeric_cols = [col for col in df_comb.columns if pd.api.types.is_numeric_dtype(df_comb[col])]

    # Get and drop countries (and indices) that have no values for any one column
    null_countries = []
    index_null = []
    g = df_comb.groupby('Country name')[numeric_cols].agg('median').reset_index()  # calculate median for all numeric columns by country
    for c in df_comb['Country name'].unique():
        c_values = g[g['Country name'] == c].values  # Numpy array of values
        if not pd.Series(c_values.reshape(c_values.shape[1])).isna().sum() == 0:  # if all values are null for an indicator in a country
            null_countries.append(c)  # append the country name to list: null_countries
    for i, row in df_comb.iterrows():  # getting the indices of all the countries in null_countries
        if row['Country name'] in null_countries:
            index_null.append(i)
    df_comb.drop(columns=list(df_comb.columns[12:]), index=index_null, inplace=True)  # drops all the rows of the countries in null countries based on their index as well as the duplicate "_y" columns
    df_comb.columns = [col[:-2] if col[-2:] == '_x' else col for col in df_comb.columns]  # removing "_x" from column name
    # print(null_cols) : Sanity check

    # Add regions manually for countries with region missing
    no_region = df_comb[df_comb['Regional indicator'].isna()]['Country name'].unique().tolist()
    df_region = pd.DataFrame({'country': no_region,  # region data found from reference dataframe
                             'region': ["Sub-Saharan Africa", "Latin America and Caribbean", "South Asia", "Sub-Saharan Africa", "Sub-Saharan Africa", "Sub-Saharan Africa", "Latin America and Caribbean", "Middle East and North Africa", "Middle East and North Africa", "Latin America and Caribbean", "Middle East and North Africa", "Latin America and Caribbean"]})
    df_region.set_index('country', inplace=True)
    for c in df_comb['Country name'].unique():
        if c in no_region:
            df_comb['Regional indicator new'] = df_comb[df_comb['Country name'].isin(no_region)]['Country name'].apply(lambda x: domain_region(x, df_region))  # apply domain transformation to countries with no region

    # Clean region column
    df_comb[['Regional indicator', 'Regional indicator new']] = df_comb[['Regional indicator', 'Regional indicator new']].astype('str')
    df_comb['Region'] = df_comb[['Regional indicator', 'Regional indicator new']].agg(', '.join, axis=1)  # join the two separate region columns into one complete column
    df_comb.Region = df_comb.Region.str.replace(', nan', '')
    df_comb.Region = df_comb.Region.str.replace('nan, ', '')
    df_comb.drop(columns=['Regional indicator', 'Regional indicator new'], inplace=True)
    # print(df_comb[df_comb['Region'].isna()]['Country name'].unique().tolist()) : Sanity check

    # Clean column names for further use
    clean_names = []
    for col in df_comb.columns:
        clean_names.append(col.replace(" ", "_").lower())  # removing spaces and replacing them with an underscore for column names, and then making them all lowercase
    df_comb.columns = clean_names
    df_comb.rename(columns={'life_ladder': 'happiness'}, inplace=True)

    # Save DataFrame to csv
    # df_comb.to_csv(os.path.join('drive', 'MyDrive', 'MA 346 Project 1', 'world-happiness-report-clean.csv'))

    return df_comb


def display(pg, df):
    if pg == 'Home':
        home()
    elif pg == 'EDA':
        eda(df)
    elif pg == 'Social Support':
        create_line_scatter(df, 'social_support')
    elif pg == 'GDP':
        create_line_scatter(df, 'log_gdp_per_capita')
    elif pg == 'Life Expectancy':
        create_line_scatter(df, 'healthy_life_expectancy_at_birth')
    elif pg == 'Happiness':
        happy(df)


def home():
    st.title('Welcome')


def eda(df):
    df.hist(figsize = (10,10))
    st.pyplot()

    # heat map in plotly
    vals = df.loc[:, df.columns != 'year']  # getting quanitative data besides the year column
    fig = px.imshow(vals.corr(), text_auto=True, color_continuous_scale='RdBu_r', title='Correlations between Happiness Indicators')  # vals.corr() takes correlations of selected columns
    st.plotly_chart(fig)

    # count of countries in plotly
    g = df.groupby('region')['country_name'].agg('unique').reset_index()  # Ensure we do not have duplicate country names
    data = g.groupby('region').agg(lambda x: x.apply(len).sum()).reset_index()  # Groupby region to get number of countries in each region
    num_countries = pd.DataFrame(data).reset_index()  # Make the groupby a dataframe and reset index to make each column a variable

    fig = px.bar(num_countries, x='region', y='country_name', title='Number of Countries per Region', text_auto=True)

    st.plotly_chart(fig)

    # print(df.describe().T)  # simple statistics for data

    num_cols = ['social_support', 'freedom_to_make_life_choices', 'perceptions_of_corruption', 'positive_affect', 'negative_affect']  # columns that are on 0-1 scale
    fig = px.box(df.loc[:, num_cols], title='Distributions for Predictors (with 0-1 scale)')
    st.plotly_chart(fig)

    fig = px.box(df, y='generosity', title='Distribution of Generosity')  # Generosity not on 0-1 scale
    st.plotly_chart(fig)

    st.write('Linear Regression Summary')
    mod = smf.ols("happiness ~ log_gdp_per_capita + social_support + healthy_life_expectancy_at_birth + freedom_to_make_life_choices + generosity + perceptions_of_corruption + positive_affect + negative_affect",
                  data=df).fit()
    st.write(mod.summary())


def create_line_scatter(df, col):
    sub = df[['year', 'region', col]]
    group_avg = pd.DataFrame(sub.groupby(['year', 'region'])[col].agg('mean').reset_index())
    fig = px.line(group_avg, x='year', y=col, color='region', markers=True, title=f'Average {" ".join(pd.Series(col.split("_")).apply(lambda x: x.capitalize()).tolist())} per Year and Region', labels={'year': 'Year', col: f'{" ".join(pd.Series(col.split("_")).apply(lambda x: x.capitalize()).tolist())}'})
    st.plotly_chart(fig)

    year = st.selectbox('Choose the year', ['No filter'] + sorted(list(df.year.unique())))
    title = f'Happiness by {" ".join(pd.Series(col.split("_")).apply(lambda x: x.capitalize()).tolist())} by Region'
    if year != 'No filter':
        df = df.loc[df.year == year, :]
        title = f'Happiness by {" ".join(pd.Series(col.split("_")).apply(lambda x: x.capitalize()).tolist())} by Region for {year}'
    fig = px.scatter(df, x=col, y='happiness', color='region', hover_data=['country_name', 'year'], title=title, labels={'happiness': 'Happiness', col: f'{" ".join(pd.Series(col.split("_")).apply(lambda x: x.capitalize()).tolist())}'})
    st.plotly_chart(fig)


def happy(df):
    # grouping the data by country name to get the region and happiness levels
    avg_hap = pd.DataFrame(df.groupby('country_name')[['happiness', 'region']].agg({'happiness': 'mean', 'region': pd.Series.mode})).reset_index().sort_values('happiness', ascending=False).set_index('country_name')
    fig = px.bar(avg_hap, x=avg_hap.index, y='happiness', title='Happiness by Region', color='region')
    fig.update_layout(xaxis_categoryorder='total descending')  # allows for descending over all regions instead of region by region grouping, which is the default
    st.plotly_chart(fig)

    year = st.selectbox('Choose the year', ['No filter'] + sorted(list(df.year.unique())))
    title = 'Relationship between Happiness, Life Expectancy, and Social Support'
    if year != 'No filter':
        df = df.loc[df.year == year, :]
        title = f'Relationship between Happiness, Life Expectancy, and Social Support in {year}'
    fig = px.scatter_3d(df, x='social_support', z='happiness', y='healthy_life_expectancy_at_birth', color='region', title=title)
    fig.update_traces(marker_size=3)
    camera = dict(eye=dict(x=2, y=2, z=0.1))  # changing the view to make it different from the default
    fig.update_layout(scene_camera=camera)
    # fig.write_html(os.path.join('drive','My Drive','MA 346 Project 1','Plotly_3D_Plot_(Gen).html')) # saving graph so we can put into the report
    st.plotly_chart(fig)


def main():
    # mount()
    data1, data2 = create_path()
    df = clean_df(data1, data2)
    # df.info()
    page = st.sidebar.radio('Choose a page:', ['Home', 'EDA', 'Social Support', 'Life Expectancy', 'GDP', 'Happiness'])
    display(page, df)


main()
