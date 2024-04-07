import streamlit as st
import numpy as np
import pickle
from PIL import Image


# predicition funtion=============================================================================================
def pred(Day_of_week, Age_band_of_driver, Sex_of_driver, Educational_level, Vehicle_driver_relation,
         Driving_experience, Type_of_vehicle, Owner_of_vehicle, Service_year_of_vehicle,
         Defect_of_vehicle, Area_accident_occured, Lanes_or_Medians, Road_allignment,
         Types_of_Junction, Road_surface_type, Road_surface_conditions, Light_conditions,
         Weather_conditions, Type_of_collision, Number_of_vehicles_involved,
         Number_of_casualties, Vehicle_movement, Casualty_class, Sex_of_casualty,
         Age_band_of_casualty, Casualty_severity, Work_of_casuality, Fitness_of_casuality,
         Pedestrian_movement, Cause_of_accident, Hour_of_Day, pipe):
    # Your prediction code here
    features = np.array([[Day_of_week, Age_band_of_driver, Sex_of_driver, Educational_level, Vehicle_driver_relation,
                          Driving_experience, Type_of_vehicle, Owner_of_vehicle, Service_year_of_vehicle,
                          Defect_of_vehicle, Area_accident_occured, Lanes_or_Medians, Road_allignment,
                          Types_of_Junction, Road_surface_type, Road_surface_conditions, Light_conditions,
                          Weather_conditions, Type_of_collision, Number_of_vehicles_involved,
                          Number_of_casualties, Vehicle_movement, Casualty_class, Sex_of_casualty,
                          Age_band_of_casualty, Casualty_severity, Work_of_casuality, Fitness_of_casuality,
                          Pedestrian_movement, Cause_of_accident, Hour_of_Day]])

    # Assuming pipe is your trained model pipeline
    results = pipe.predict(features)
    return results


# create streamlit app==================================================================================================
# load pipe and image
pipe = pickle.load(open('10 pipe.pkl','rb'))
img = Image.open('10 pipe img.png')



st.title('Accident Severity Prediction With Sklearn Pipeline...')
# Display the image
st.image(img, use_column_width=True)


# Sidebar for user inputs
with st.sidebar:
    st.write('### User Inputs')
    Day_of_week = st.selectbox('Day of Week',
                               ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    Age_band_of_driver = st.selectbox('Age Band of Driver',
                                      ['17-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-65',
                                       '66-75', 'Over 75'])
    Sex_of_driver = st.selectbox('Sex of Driver', ['Male', 'Female', 'Unknown'])
    Educational_level = st.selectbox('Educational Level', ['Above high school', 'Junior high school', 'Elementary school',
                                                           'High school', 'Unknown', 'Illiterate', 'Writing & reading'])
    Vehicle_driver_relation = st.selectbox('Vehicle Driver Relation', ['Employee', 'Unknown', 'Owner', 'Other'])
    Driving_experience = st.selectbox('Driving Experience', ['1-2yr', 'Above 10yr', '5-10yr', '2-5yr', 'No Licence',
                                                             'Below 1yr', 'unknown'])
    Type_of_vehicle = st.selectbox('Type of Vehicle', ['Automobile', 'Public (> 45 seats)', 'Lorry (41?100Q)', 'Public (13?45 seats)',
                                                        'Lorry (11?40Q)', 'Long lorry', 'Public (12 seats)', 'Taxi', 'Pick up upto 10Q',
                                                        'Stationwagen', 'Ridden horse', 'Other', 'Bajaj', 'Turbo', 'Motorcycle',
                                                        'Special vehicle', 'Bicycle'])
    Owner_of_vehicle = st.selectbox('Owner of Vehicle', ['Owner', 'Governmental', 'Organization', 'Other'])
    Service_year_of_vehicle = st.selectbox('Service Year of Vehicle', ['Above 10yr', '5-10yrs', '1-2yr', '2-5yrs', 'Unknown', 'Below 1yr'])
    Defect_of_vehicle = st.selectbox('Defect of Vehicle', ['No defect', '7', '5'])
    Area_accident_occured = st.selectbox('Area of Accident Occurred', ['Residential areas', 'Office areas', 'Recreational areas',
                                                                       'Industrial areas', 'Other', 'Church areas', 'Market areas',
                                                                       'Unknown', 'Rural village areas', 'Outside rural areas',
                                                                       'Hospital areas', 'School areas',
                                                                       'Rural village areasOffice areas', 'Recreational areas'])
    Lanes_or_Medians = st.selectbox('Lanes or Medians', ['Undivided Two way', 'other', 'Double carriageway (median)', 'One way',
                                                         'Two-way (divided with solid lines road marking)',
                                                         'Two-way (divided with broken lines road marking)', 'Unknown'])
    Road_allignment = st.selectbox('Road Allignment', ['Tangent road with flat terrain', 'Tangent road with mild grade and flat terrain',
                                                       'Escarpments', 'Tangent road with rolling terrain', 'Gentle horizontal curve',
                                                       'Tangent road with mountainous terrain and', 'Steep grade downward with mountainous terrain',
                                                       'Sharp reverse curve', 'Steep grade upward with mountainous terrain'])
    Types_of_Junction = st.selectbox('Types of Junction', ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other', 'Unknown', 'T Shape',
                                                           'X Shape'])
    Road_surface_type = st.selectbox('Road Surface Type', ['Asphalt roads', 'Earth roads', 'Asphalt roads with some distress',
                                                           'Gravel roads', 'Other'])
    Road_surface_conditions = st.selectbox('Road Surface Conditions', ['Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'])
    Light_conditions = st.selectbox('Light Conditions', ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
                                                         'Darkness - lights unlit'])
    Weather_conditions = st.selectbox('Weather Conditions', ['Normal', 'Raining', 'Raining and Windy', 'Cloudy', 'Other', 'Windy',
                                                             'Snow', 'Unknown', 'Fog or mist'])
    Type_of_collision = st.selectbox('Type of Collision', ['Collision with roadside-parked vehicles', 'Vehicle with vehicle collision',
                                                           'Collision with roadside objects', 'Collision with animals', 'Other',
                                                           'Rollover', 'Fall from vehicles', 'Collision with pedestrians',
                                                           'With Train', 'Unknown'])
    Number_of_vehicles_involved = st.number_input('Number of Vehicles Involved', min_value=1, max_value=10, step=1, value=1)
    Number_of_casualties = st.number_input('Number of Casualties', min_value=1, max_value=10, step=1, value=1)
    Vehicle_movement = st.selectbox('Vehicle Movement', ['Going straight', 'U-Turn', 'Moving Backward', 'Turnover', 'Waiting to go',
                                                         'Getting off', 'Reversing', 'Unknown', 'Parked', 'Stopping', 'Overtaking',
                                                         'Other', 'Entering a junction'])
    Casualty_class = st.selectbox('Casualty Class', ['na', 'Driver or rider', 'Pedestrian', 'Passenger'])
    Sex_of_casualty = st.selectbox('Sex of Casualty', ['na', 'Male', 'Female'])
    Age_band_of_casualty = st.selectbox('Age Band of Casualty', ['na', '31-50', '18-30', 'Under 18', 'Over 51', '5'])
    Casualty_severity = st.selectbox('Casualty Severity', ['na', '3', '2', '1'])
    Work_of_casuality = st.selectbox('Work of Casualty', ['Driver', 'Other', 'Unemployed', 'Employee', 'Self-employed', 'Student', 'Unknown'])
    Fitness_of_casuality = st.selectbox('Fitness of Casualty', ['Normal', 'Deaf', 'Other', 'Blind', 'NormalNormal'])
    Pedestrian_movement = st.selectbox('Pedestrian Movement', ["Not a Pedestrian",
                                                               "Crossing from driver's nearside",
                                                               "Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle",
                                                               "Unknown or other",
                                                               "Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle",
                                                               "In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)",
                                                               "Walking along in carriageway, back to traffic",
                                                               "Walking along in carriageway, facing traffic",
                                                               "In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle"])
    Cause_of_accident = st.selectbox('Cause of Accident', ['Moving Backward', 'Overtaking', 'Changing lane to the left',
                                                           'Changing lane to the right', 'Overloading', 'Other',
                                                           'No priority to vehicle', 'No priority to pedestrian',
                                                           'No distancing', 'Getting off the vehicle improperly',
                                                           'Improper parking', 'Overspeed', 'Driving carelessly',
                                                           'Driving at high speed', 'Driving to the left', 'Unknown',
                                                           'Overturning', 'Turnover', 'Driving under the influence of drugs',
                                                           'Drunk driving'])
    Hour_of_Day = st.selectbox('Hour of Day', [17, 1, 14, 22, 8, 15, 12, 18, 13, 20, 16, 21, 9, 10, 19, 11, 23, 7, 0, 5, 6, 4, 3, 2])



# Predict button
if st.sidebar.button('Predict'):
    predicted_class = pred(Day_of_week=Day_of_week,
                           Age_band_of_driver=Age_band_of_driver,
                           Sex_of_driver=Sex_of_driver,
                           Educational_level=Educational_level,
                           Vehicle_driver_relation=Vehicle_driver_relation,
                           Driving_experience=Driving_experience,
                           Type_of_vehicle=Type_of_vehicle,
                           Owner_of_vehicle=Owner_of_vehicle,
                           Service_year_of_vehicle=Service_year_of_vehicle,
                           Defect_of_vehicle=Defect_of_vehicle,
                           Area_accident_occured=Area_accident_occured,
                           Lanes_or_Medians=Lanes_or_Medians,
                           Road_allignment=Road_allignment,
                           Types_of_Junction=Types_of_Junction,
                           Road_surface_type=Road_surface_type,
                           Road_surface_conditions=Road_surface_conditions,
                           Light_conditions=Light_conditions,
                           Weather_conditions=Weather_conditions,
                           Type_of_collision=Type_of_collision,
                           Number_of_vehicles_involved=Number_of_vehicles_involved,
                           Number_of_casualties=Number_of_casualties,
                           Vehicle_movement=Vehicle_movement,
                           Casualty_class=Casualty_class,
                           Sex_of_casualty=Sex_of_casualty,
                           Age_band_of_casualty=Age_band_of_casualty,
                           Casualty_severity=Casualty_severity,
                           Work_of_casuality=Work_of_casuality,
                           Fitness_of_casuality=Fitness_of_casuality,
                           Pedestrian_movement=Pedestrian_movement,
                           Cause_of_accident=Cause_of_accident,
                           Hour_of_Day=Hour_of_Day,
                           pipe=pipe)

    # Display prediction result
    if predicted_class[0] == 2:
        st.write("Predicted Injury: Slight Injury")
    elif predicted_class[0] == 1:
        st.write("Predicted Injury: Serious Injury")
    else:
        st.write("Predicted Injury: Fatal Injury")
