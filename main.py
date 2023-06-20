
import streamlit as st
import plotly.express as px
import datetime
import plotly.subplots as sp
import plotly.graph_objects as go
import requests
import pandas as pd
import numpy as np

import json





class Project:
    def __init__(self, saleDate):
     self.saleDate1 = saleDate
    def set_ownerName(self, value):
        self.ownername = value


    def get_ownerName(self):
        return self.ownername

    def set_saleDate(self, value):
        self.saleDate1 = value


    def get_saleDate(self):
        return self.saleDate1

    def set_FLADate(self, value):
        self.FLAdate = value


    def get_FLADate(self):
        return self.FLAdate

    def set_welcomeDate(self, value):
        self.welcomeCallCompleteDate2 = value


    def get_welcomeDate(self):
        return self.welcomeCallCompleteDate2

    def set_installer(self, value):
        self.installer = value


    def get_installer(self):
        return self.installer
    def set_qualityCheckDate(self, value):
        self.qualityCheckDate5 = value


    def get_qualityCheckDate(self):
        return self.qualityCheckDate5

    def set_siteSurveyComplete(self, value):
        self.siteSurveyCompleteDate3 = value


    def get_siteSurveyComplete(self):
         return self.siteSurveyComplete3

    def set_NTPDate(self, value):
        self.NTPDate4 = value


    def get_NTPDate(self):
        return self.NTPDate4


    def set_solarPlans(self, value):
        self.solarPlans6 = value

    def get_solarPlans(self):
        return self.solarPlans6


    def set_solarInstall(self, value):
        self.solarInstall7 = value


    def get_solarPermit(self):
        return self.solarPermit


    def set_solarPermit(self, value):
        self.solarPermit = value


    def get_solarinstall(self):
        return self.solarInstall7


    def set_FinalInspection(self, value):
        self.FinalInspection = value


    def get_FinalInspection(self):
        return self.FinalInspection


    def set_PTO(self, value):
        self.PTO = value


    def get_PTO(self):
        return self.PTO


    def set_Address(self, value):
        self.address = value


    def get_Address(self):
        return self.address


    def set_Completed(self, value):
        self.Completed = value


    def get_Compelted(self):
        return self.Completed






# Set the base URL for the QuickBase API
base_url = "https://api.quickbase.com/v1"


# Setting QuickBase realm and application tokens
realm = "voltaic.quickbase.com"
app_token = "b7738j_qjt3_0_dkaew43bvzcxutbu9q4e6crw3ei3"


# Setting the API endpoint for the desired QuickBase table
table_id = "your_table_id"




api_endpoint = f"/reports/520/run?tableId=br5cqr4r3"


# Setting the headers with the required authentication and content type
headers = {
"QB-Realm-Hostname": realm,
"Authorization": f"QB-USER-TOKEN {app_token}",
"Content-Type": "application/json"
}


# Setting the data to be sent in the request
# data = {
# "fields": {
# "Field1": "Value1",
# "Field2": "Value2",
# "Field3": "Value3"
# }
# }


# Sending a POST request to create a new record in the QuickBase table
response = requests.post(f"{base_url}{api_endpoint}", headers=headers)


# Checking the response status code
if response.status_code == 200:
   # st.write("Loaded Volatic Data Successfully!")
    records_json = response.json()
    # Creating a list to hold project models
    project_models = []


    parsed_response = json.loads(response.text)
    pretty_response = json.dumps(parsed_response, indent=4)
    # print(pretty_response)
    try:
        for record in records_json.get("data", []):
            #                                                                                                                                    print("RECORD!")
            # print(record)
            homeownerName = record.get("105").get("value")
            saleDate = record.get("102").get("value")
            address = record.get("92").get("value")
            welcomeDate = record.get("1377").get("value")
            siteSurveyDate = record.get("1641").get("value")
            NTPDate = record.get("862").get("value")
            FLADate = record.get("1654").get("value")
            QualityControlCheckDate = record.get("856").get("value")
            SolarPlansDate = record.get("1650").get("value")
            SolarPermitDate = record.get("1656").get("value")
            SolarInstallDate = record.get("877").get("value")
            FinalInspectionDate = record.get("883").get("value")
            PTODate = record.get("1670").get("value")
            instaler = record.get("634").get("value")
            CompletedDate = record.get("387").get("value")
            #
            # print(saleDate)
            # print(welcomeDate)




            # Create project model and add it to the list


            project_model = Project(saleDate)
            project_model.set_FLADate(FLADate)
            project_model.set_ownerName(homeownerName)
            project_model.set_welcomeDate(welcomeDate)
            project_model.set_siteSurveyComplete(siteSurveyDate)
            project_model.set_NTPDate(NTPDate)
            project_model.set_qualityCheckDate(QualityControlCheckDate)
            project_model.set_solarPlans(SolarPlansDate)
            project_model.set_solarPermit(SolarPermitDate)
            project_model.set_solarInstall(SolarInstallDate)
            project_model.set_installer(instaler)
            project_model.set_Address(address)
            project_model.set_FinalInspection(FinalInspectionDate)
            project_model.set_PTO(PTODate)
            project_model.set_Completed(CompletedDate)
            project_models.append(project_model)




        # Convert project models into a DataFrame
        data = {
        "homeownerName": [project.ownername for project in project_models],
        "installer": [project.installer for project in project_models],
        "address": [project.address for project in project_models],
        "SaleDate1": [project.saleDate1 for project in project_models],
        "WelcomeCallDate2": [project.welcomeCallCompleteDate2 for project in project_models],
        "siteSurveyDate3": [project.siteSurveyCompleteDate3 for project in project_models],
        "noticeToPermitDate4": [project.NTPDate4 for project in project_models],
        "QualityCheckDate5": [project.qualityCheckDate5 for project in project_models],
        "FLADate6": [project.FLAdate for project in project_models],
        "SolarPlansReceivedDate6": [project.solarPlans6 for project in project_models],
        "SolarPermitReceiveDate7": [project.solarPermit for project in project_models],
        "SolarInstallCompleteDate8": [project.solarInstall7 for project in project_models],
        "FinalInspectionDate9": [project.FinalInspection for project in project_models],
        "PermissionToOperateApprovedDate10": [project.PTO for project in project_models],


        }
        df = pd.DataFrame(data)


        # Convert date columns to datetime format
        date_cols = [ "SaleDate1", "WelcomeCallDate2", "siteSurveyDate3", "noticeToPermitDate4", "QualityCheckDate5","FLADate6",
        "SolarPlansReceivedDate6", "SolarPermitReceiveDate7", "SolarInstallCompleteDate8", "FinalInspectionDate9", "PermissionToOperateApprovedDate10",
        ]


        df[date_cols] = df[date_cols].apply(pd.to_datetime)


        # Calculate duration for each stage

        #CREATE NEW DURATION COLUMNS FOR DURATION DF
        duration_cols = []
        for i in range(1, len(date_cols)):

            columnName = "0"
            if i == 0:
             columnName = "Voltaic Check"
            elif i==1:
             columnName = "Welcome"
            elif i==2:
                columnName = "SS"
            elif i==3:
                columnName = "NTP"
            elif i==4:
                columnName = "QCCheck"
            elif i == 5:
                columnName = "FLA"
            elif i==6:
                columnName = "SolarPlans"
            elif i==7:
                columnName = "SolarPermit"

            elif i==8:
                columnName = "SolarInstall"
            elif i==9:
                columnName = "FinalInspection"
            elif i==10:
                columnName = "PTO"


            duration_col = f"{columnName}"
            df[duration_col] = (df[date_cols[i]] - df[date_cols[i - 1]]).dt.days
            duration_cols.append(duration_col)


        # Create a new DataFrame with durations
        new_df = df[[ "homeownerName","address", "installer"] + duration_cols]



        # Create three separate dataframes based on unique identifiers
        df_1 = df[df['installer'] == 'Voltaic Construction']
        df_2 = df[df['installer'] == 'Greenspire']
        df_3 = df[df['installer'] == 'AC/DC']
        df_4 = df[df['installer'] == 'Proclaim Solar']
        df_5 = df[df['installer'] == 'Titanium Solar']
        df_6 = df[df['installer'] == 'Energy Service Partners']



        installers_df = [df_1, df_2, df_5]




        # =====================-===========================-TESTING-===========================-===========================
        original_addresses = []
        original_homeowners = []
        original_installers = []
        original_saleDates = []

        original_welcomeDates = []
        original_siteSurveydates = []
        original_ntpdates = []
        original_qccheckDates = []
        original_FLAdates = []
        original_solarPlansDates = []
        original_SolarPermitDates = []
        original_solarInstallDates = []
        original_FinalInspectionDates = []
        original_PTODates = []


        # Create an empty DataFrame to store the combined data

        combined_df = pd.DataFrame()











    #Loop through installer to get global stats by installers
        for installer in installers_df:

            df = installer
            # first_value = df.iloc[0, 1]
            original_addresses.append(df['address'].copy())
            original_homeowners.append(df['homeownerName'].copy())
            original_installers.append(df['installer'].copy())
            original_saleDates.append(df['SaleDate1'].copy())

            original_welcomeDates.append(df['WelcomeCallDate2'].copy())
            original_siteSurveydates.append(df['siteSurveyDate3'].copy())
            original_ntpdates.append(df['noticeToPermitDate4'].copy())
            original_qccheckDates.append(df['QualityCheckDate5'].copy())
            original_FLAdates.append(df['FLADate6'].copy())
            original_solarPlansDates.append(df['SolarPlansReceivedDate6'].copy())
            original_SolarPermitDates.append(df['SolarPermitReceiveDate7'].copy())
            original_solarInstallDates.append(df['SolarInstallCompleteDate8'].copy())
            original_FinalInspectionDates.append(df['FinalInspectionDate9'].copy())
            original_PTODates.append(df['PermissionToOperateApprovedDate10'].copy())


            # Removing "installer", "homeownerName", "address" columns
            duration_df = df[duration_cols]
            # st.write("Duration Dataframe")
            # st.dataframe(duration_df)


            # Calculate column-wise averagesC
            column_averages = duration_df.mean()

            try:
                # Replace negatives with the column averages, leave zero and NaN values unchanged
                for column in duration_df.columns:
                    duration_df[column] = duration_df[column].apply(lambda x: column_averages[column] if x < 0 else x)

                # st.write(f"Avg stage duration for {column} and installer"
                # f"{original_installers[-1]}: {column_averages[column]}")
                # st.write(f'{first_value} injected Avg std deviation')
                # st.dataframe(duration_df)

                # Add original columns back to duration_df
                duration_df['installer'] = original_installers[-1]
                duration_df['address'] = original_addresses[-1]
                duration_df['homeownerName'] = original_homeowners[-1]
                duration_df['SaleDate'] = original_saleDates[-1]

                duration_df['WelcomeCallDate2'] = original_welcomeDates[-1]
                duration_df['siteSurveyDate3'] = original_siteSurveydates[-1]
                duration_df['noticeToPermitDate4'] = original_SolarPermitDates[-1]
                duration_df['QualityCheckDate5'] = original_qccheckDates[-1]
                duration_df['FLADate6'] = original_FLAdates[-1]
                duration_df['SolarPlansReceivedDate6'] = original_solarPlansDates[-1]
                duration_df['SolarPermitReceiveDate7'] = original_SolarPermitDates[-1]
                duration_df['SolarInstallCompleteDate8'] = original_solarInstallDates[-1]
                duration_df['FinalInspectionDate9'] = original_FinalInspectionDates[-1]
                duration_df['PermissionToOperateApprovedDate10'] = original_PTODates[-1]

                # Combine the duration_df with the combined_df
                combined_df = pd.concat([combined_df, duration_df])

                # st.write("Updated DataFrame:")
                # st.dataframe(duration_df)
            except Exception as e:
                print(f"An error occurred: {e}")

            finally:
                print("Completed")


            # st.dataframe(duration_df)


        # Display the combined DataFrame


        combined_df['projectDuration'] = combined_df[duration_cols].sum(axis=1)

      #  st.write("Combined DataFrame:")
       # st.dataframe(combined_df)

        # =====================-^^^===========================-TESTING-===========================-^^^===========================








#AI Stuff



        # # AI stuff
        #
        #
        # OPENAI_KEY = "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"
        #
        # llm = OpenAI(OPENAI_KEY)
        # # pandas_ai = PandasAI(llm, verbose=True, conversational=False)
        # pandas_ai = PandasAI(llm)
        # # response = pandas_ai.run(df, "What is the average project duration for each stage by installer?")
        # # print(response)
        #
        #
        # prompt = st.text_area('What do you want to know about voltaic? ')
        #
        #
        # if st.button('Generate'):
        #     if prompt:
        #         with st.spinner("Analyzing Voltaic...."):
        #             res = pandas_ai.run(combined_df, prompt=prompt)
        #         print(res)
        #         st.write(res)
        #
        #
        #     else:
        #         st.warning("Enter a prompt.")
        #










    except json.JSONDecodeError:
        print("Unable to parse response as JSON.")
    # st.write(response.text)


else:
    st.write(f"Error: {response.status_code} - {response.text}")














#=============================================


















































#=============================================================================

# Display "KPIs" and the dataframe "combined_df"
st.write("KPIs")
st.dataframe(combined_df)

# Display global average pipeline duration
st.title(":bar_chart: Global Avg Pipeline Duration")
st.write("SOLD between 01/06/23 - 06/20/23")
st.markdown("##")

# Calculate average duration for each stage
welcome_avg_completed = int(combined_df["Welcome"].mean())
ss_avg_completed = int(combined_df["SS"].mean())
ntp_avg_completed = int(combined_df["NTP"].mean())
qc_avg_completed = int(combined_df["QCCheck"].mean())
fla_avg_completed = int(combined_df["FLA"].mean())
plans_avg_completed = int(combined_df["SolarPlans"].mean())
permit_avg_completed = int(combined_df["SolarPermit"].mean())
install_avg_completed = int(combined_df["SolarInstall"].mean())
inspection_avg_completed = int(combined_df["FinalInspection"].mean())
pto_avg_completed = int(combined_df["PTO"].mean())

# Calculate global average completed duration
global_avg_completed = int(
    welcome_avg_completed + ss_avg_completed + ntp_avg_completed + qc_avg_completed +
    fla_avg_completed + permit_avg_completed + plans_avg_completed + install_avg_completed +
    inspection_avg_completed + pto_avg_completed
)
# Specify the stage columns

# Create columns for stage durations
stage1_colum, stage2_colum, stage3_colum, stage4_colum, stage5_colum, stage6_colum, stage7_colum, stage8_colum, stage9_colum, stage10_colum, stage11_colum = st.columns(11)

# Display stage durations in the columns
with stage1_colum:
    st.markdown("###### Welcome")
    st.subheader(welcome_avg_completed)
with stage2_colum:
    st.markdown("##### SS")
    st.subheader(ss_avg_completed)
with stage3_colum:
    st.markdown("##### NTP")
    st.subheader(ntp_avg_completed)
with stage4_colum:
    st.markdown("##### QC")
    st.subheader(qc_avg_completed)
with stage5_colum:
    st.markdown("##### FLA")
    st.subheader(fla_avg_completed)
with stage6_colum:
    st.markdown("##### Permit")
    st.subheader(permit_avg_completed)
with stage7_colum:
    st.markdown("##### Plans")
    st.subheader(plans_avg_completed)
with stage8_colum:
    st.markdown("##### Install")
    st.subheader(install_avg_completed)
with stage9_colum:
    st.markdown("##### FI")
    st.subheader(inspection_avg_completed)
with stage10_colum:
    st.markdown("##### PTO")
    st.subheader(pto_avg_completed)
with stage11_colum:
    st.markdown("##### Total Avg")
    st.subheader(global_avg_completed)

# =============== ==================    Sidebar for filters  ===
st.sidebar.header("Filter")


# Define all date fields with labels
date_fields = {
    'SaleDate': 'Sale Date',
    'WelcomeCallDate2': 'Welcome Call',
    'siteSurveyDate3': 'Site Survey',
    'noticeToPermitDate4': 'NTP',
    'QualityCheckDate5': 'QC Check',
    'FLADate6': 'FLA Date',
    'SolarPlansReceivedDate6': 'Solar Plans',
    'SolarPermitReceiveDate7': 'Solar Permit',
    'SolarInstallCompleteDate8': 'Solar Install',
    'FinalInspectionDate9': 'Final Inspection',
    'PermissionToOperateApprovedDate10': 'PTO'
}

# Find the absolute min and max across all fields
abs_min_date = min(combined_df[field].dropna().min() for field in date_fields).date()
abs_max_date = max(combined_df[field].dropna().max() for field in date_fields).date()

# Global date range selection
selected_date = st.sidebar.date_input("Select a common date range:", [abs_min_date, abs_max_date])
selected_dates_datetime = [pd.to_datetime(date) for date in selected_date]

# List to store selected fields
selected_fields = []

# Iterate over each date field and create checkboxes for filtering
for field, label in date_fields.items():
    if st.sidebar.checkbox(f"Completed {label} "):
        selected_fields.append(field)

# Create an initial boolean mask with all False values
filter_mask = pd.Series([False] * len(combined_df), index=combined_df.index)

# Single slider for number of days before and after
num_days = st.sidebar.slider("Select the number of days before/after:", 0, 365, 0)

# Apply filters based on selected date range and fields
for field in selected_fields:
    start_date = selected_dates_datetime[0] - pd.Timedelta(days=num_days)
    end_date = selected_dates_datetime[1] + pd.Timedelta(days=num_days)
    current_mask = combined_df[field].dropna().isna() | ((combined_df[field] >= start_date) & (combined_df[field] <= end_date))
    filter_mask = filter_mask | current_mask

# Apply the combined filter mask to the original DataFrame
filtered_df = combined_df[filter_mask]

# Rest of the code...

# Continue with the rest of your code...

# Multiselect for installer
installerFilter = st.sidebar.multiselect(
    "Select the installer:",
    options=filtered_df['installer'].unique().tolist(),
    default=filtered_df['installer'].unique().tolist()
)

installerquery = filtered_df.query("installer == @installer")

filtered_df = filtered_df.query("installer == @installerFilter")

st.write("Filtered DF")
st.dataframe(filtered_df)




























# ========================  ----MAIN PAGE ------ ==========================

# Extract the selected date range
selected_date_fields = [f"**{date_fields[field]}**" for field in selected_fields]
selected_start_date = selected_dates_datetime[0].strftime("%Y-%m-%d")
selected_end_date = selected_dates_datetime[1].strftime("%Y-%m-%d")

title = f":bar_chart: Cumulative average project duration for projects that have completed {', '.join(selected_date_fields)} between {selected_start_date} to {selected_end_date}"
st.title(title)
st.markdown(f"### between: {selected_start_date} to {selected_end_date}")







# Compute duration by installer
duration_by_installer = filtered_df.groupby(by=["installer"]).mean()[["projectDuration"]].sort_values(by="projectDuration")

# Create the bar chart using Plotly Express
fig_duration_by_installer = px.bar(
    duration_by_installer,
    x="projectDuration",
    y=duration_by_installer.index,
    orientation="h",
    title="<b>Duration by installer</b>",
    color_discrete_sequence=["#0083b8"] * len(duration_by_installer),
    template="plotly_white",
    opacity=0.8,  # Adjust the opacity of the bars
)

# Add yellow line for target duration
fig_duration_by_installer.add_shape(
    type="line",
    x0=60,  # Target duration
    y0=-0.5,
    x1=60,
    y1=len(duration_by_installer)-0.5,
    line=dict(color="yellow", width=3),
)

# Show the figure
st.plotly_chart(fig_duration_by_installer)









# Calculate the average duration for each stage and installer category
stage_columns = ['Welcome', 'SS', 'NTP', 'QCCheck', 'FLA', 'SolarPlans', 'SolarPermit', 'SolarInstall', 'FinalInspection', 'PTO']
average_duration = filtered_df.groupby('installer')[stage_columns].mean().reset_index()

# Reshape the data to long format for the stacked bar chart
average_duration_long = pd.melt(average_duration, id_vars='installer', value_vars=stage_columns, var_name='Stage', value_name='Average Duration')

# Create the stacked bar chart using Plotly Express
fig = px.bar(average_duration_long, x='Stage', y='Average Duration', color='installer', barmode='stack', title='Pipeline Breakdown by Installer')

# Add yellow lines for target durations
target_durations = [5, 6, 7, 8, 9, 6, 5, 8, 5, 4]  # Target durations for each stage

if len(target_durations) == len(stage_columns):
    for i, stage in enumerate(stage_columns):
        fig.add_shape(
            type="line",
            x0=target_durations[i],
            y0=-0.5,
            x1=target_durations[i],
            y1=1,
            line=dict(color="yellow", width=3),
        )

# Display the chart
st.plotly_chart(fig)










































# Calculate the average duration for each stage
welcome_avg_completed = filtered_df["Welcome"]
ss_avg_completed = filtered_df["SS"]
ntp_avg_completed = filtered_df["NTP"]
qc_avg_completed = filtered_df["QCCheck"]
fla_avg_completed = filtered_df["FLA"]
plans_avg_completed = filtered_df["SolarPlans"]
permit_avg_completed = filtered_df["SolarPermit"]
install_avg_completed = filtered_df["SolarInstall"]
inspection_avg_completed = filtered_df["FinalInspection"]
pto_avg_completed = filtered_df["PTO"]

# Check for NaN values and assign a default value
default_value = 0

welcome_avg_completed = int(np.nanmean(welcome_avg_completed)) if not np.isnan(np.nanmean(welcome_avg_completed)) else default_value
ss_avg_completed = int(np.nanmean(ss_avg_completed)) if not np.isnan(np.nanmean(ss_avg_completed)) else default_value
ntp_avg_completed = int(np.nanmean(ntp_avg_completed)) if not np.isnan(np.nanmean(ntp_avg_completed)) else default_value
qc_avg_completed = int(np.nanmean(qc_avg_completed)) if not np.isnan(np.nanmean(qc_avg_completed)) else default_value
fla_avg_completed = int(np.nanmean(fla_avg_completed)) if not np.isnan(np.nanmean(fla_avg_completed)) else default_value
plans_avg_completed = int(np.nanmean(plans_avg_completed)) if not np.isnan(np.nanmean(plans_avg_completed)) else default_value
permit_avg_completed = int(np.nanmean(permit_avg_completed)) if not np.isnan(np.nanmean(permit_avg_completed)) else default_value
install_avg_completed = int(np.nanmean(install_avg_completed)) if not np.isnan(np.nanmean(install_avg_completed)) else default_value
inspection_avg_completed = int(np.nanmean(inspection_avg_completed)) if not np.isnan(np.nanmean(inspection_avg_completed)) else default_value
pto_avg_completed = int(np.nanmean(pto_avg_completed)) if not np.isnan(np.nanmean(pto_avg_completed)) else default_value

global_avg_completed = int(
    welcome_avg_completed + ss_avg_completed + ntp_avg_completed + qc_avg_completed +
    fla_avg_completed + permit_avg_completed + plans_avg_completed + install_avg_completed +
    inspection_avg_completed + pto_avg_completed
)
with stage1_colum:
    st.markdown("###### Welcome")
    st.subheader(welcome_avg_completed)
with stage2_colum:
    st.markdown("##### SS")
    st.subheader(ss_avg_completed)
with stage3_colum:
    st.markdown("##### NTP")
    st.subheader(ntp_avg_completed)
with stage4_colum:
    st.markdown("##### QC")
    st.subheader(qc_avg_completed)
with stage5_colum:
    st.markdown("##### FLA")
    st.subheader(fla_avg_completed)
with stage6_colum:
    st.markdown("##### Permit")
    st.subheader(permit_avg_completed)
with stage7_colum:
    st.markdown("##### Plans")
    st.subheader(plans_avg_completed)
with stage8_colum:
    st.markdown("##### Install")
    st.subheader(install_avg_completed)
with stage9_colum:
    st.markdown("##### FI")
    st.subheader(inspection_avg_completed)
with stage10_colum:
    st.markdown("##### PTO")
    st.subheader(pto_avg_completed)
with stage11_colum:
    st.markdown("##### Total Avg")
    st.subheader(global_avg_completed)






# ===================== WATERFALL GRAPH ===========================


# Calculate the average duration for each stage and installer category
average_duration = filtered_df.groupby(['installer'])[stage_columns].mean().reset_index()

# Create a list of installer names
installers = average_duration['installer'].tolist()

# Create an empty list to store the waterfall traces for each installer
waterfall_traces = []

# Iterate over each installer
for installer in installers:
    # Get the average durations for the installer
    installer_durations = average_duration.loc[average_duration['installer'] == installer, stage_columns].values[
        0].tolist()

    # Calculate the cumulative durations
    cumulative_durations = [0] + [sum(installer_durations[:i + 1]) for i in range(len(installer_durations))]

    # Create the waterfall trace for the installer
    waterfall_trace = go.Waterfall(
        x=stage_columns,
        y=installer_durations,
        measure=['absolute'] + ['relative'] * (len(installer_durations) - 1),
        textposition='auto',
        texttemplate='%{y:.0f}',
        name=installer,
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
        increasing={'marker': {'color': '#0083b8'}},
        decreasing={'marker': {'color': '#ff4d4d'}}
    )

    # Append the trace to the list
    waterfall_traces.append(waterfall_trace)

# Create the layout
layout = go.Layout(
    title='<b>CRM Pipeline Duration Waterfall Chart by Installer</b>',
    xaxis=dict(title='Stages'),
    yaxis=dict(title='Duration (days)'),
)

# Create the figure
fig_waterfall = go.Figure(data=waterfall_traces, layout=layout)

# Display the waterfall chart
st.plotly_chart(fig_waterfall)