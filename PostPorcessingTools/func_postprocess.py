
import numpy as np
import matplotlib.pyplot as plt
import os
def ang_encounter(ang_ship,ang_wave):
    if(ang_wave>ang_ship):
        theta=ang_wave-ang_ship
        return theta
    else:
        theta=360-(ang_ship-ang_wave)
        return theta
def dist_arc(loni,lati,lone,late):   #resultat en radiants    
    cosp=(np.cos(np.deg2rad(90-lati))*np.cos(np.deg2rad(90-late)) +
        np.sin(np.deg2rad(90-lati))*np.sin(np.deg2rad(90-late)) * 
        np.cos(np.deg2rad(lone-loni)))    
    return np.arccos(cosp)   #np.arccos(cosp)

def dist_mn(loni,lati,lone,late):                   
    d_rads=dist_arc(loni,lati,lone,late)
    d=d_rads*180/np.pi*60
    return d
            
def rumIni(loni,lati,lone,late):
    if lati==late:
        if loni >lone:
            return 270
        else:
            return 90     
    loni=loni+0.00001    
    k=dist_arc(loni,lati,lone,late)
    cosI=(np.cos(np.deg2rad(90-late))- np.cos(k)*
          np.cos(np.deg2rad(90-lati))) /((np.sin(k)) *
          np.sin(np.deg2rad(90-lati)) ) 
    I=np.arccos(cosI)
    I=I*180/np.pi
    if loni>lone:
        return  360-I
    else:
        return I

def rumEnd(loni,lati,lone,late):
    if lati==late:
        if loni >lone:
            return 270
        else:
            return 90
    loni=loni+0.00001        
    k=dist_arc(loni,lati,lone,late)
    cosE=(np.cos(np.deg2rad(90-lati))- np.cos(k)*
          np.cos(np.deg2rad(90-late))) /((np.sin(k)) *
          np.sin(np.deg2rad(90-late)) ) 
    E=np.arccos(cosE)
    E=E*180/np.pi
    if loni>lone:
        return  180+E
    else:
        return 180 -E
def distL(Lst,m):    
    ''' Last llista,  m valor donat la funcio torna
    el index de l'element de la llista mes prox a m
    '''
    ele=-1
    if m==0:
        return 0
    if Lst[-1]<m:
        return len(Lst)-1
    for i in range(len(Lst)):
        if (Lst[i]-m) < 0: # pssa el valor buscat
            ele=i
    d1=m-Lst[ele]
    d2=Lst[ele+1] -m
    if d1<d2 :
        return ele
    else:
        if ele>=len(Lst):
            return ele
        return ele+1  
def process_npz_to_txt(npz_file_path, output_file_path):
    """
    Reads data from an .npz file and writes it to a text file.

    Args:
        npz_file_path (str): Path to the input .npz file.
        output_file_path (str): Path to the output .txt file.
    """
    # Load the .npz file
    data = np.load(npz_file_path)

    # Open the output text file
    with open(output_file_path, 'w') as f:
        # Iterate through the arrays in the .npz file
        for key in data.files:
            f.write(f"Key: {key}\n")  # Write the key
            f.write("Data:\n")
            np.savetxt(f, data[key], fmt='%s')  # Write the array data in text format
            f.write("\n")  # Add a blank line between entries

    print(f"Data has been successfully saved to {output_file_path}.")

# Example usage:
npz_file = r'C:\Users\praveena\OneDrive\Desktop\BASE PAPER\Ship-Routing\Ship-routing\in\waves.npz'  # Replace with the path to your .npz file
output_file = 'data.txt'
process_npz_to_txt(npz_file, output_file)


def roc_curve():
    # Simulated data for plotting (ensure these values match your actual model evaluation metrics)
    fpr_nn = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]  # False positive rates for Neural Network
    tpr_nn = [0.0, 0.5, 0.6, 0.7, 0.85, 1.0]  # True positive rates for Neural Network
    roc_auc_nn = 0.90  # AUC for Neural Network

    fpr_nn_rl = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0]  # False positive rates for NN + RL
    tpr_nn_rl = [0.0, 0.6, 0.75, 0.85, 0.95, 1.0]  # True positive rates for NN + RL
    roc_auc_nn_rl = 0.97  # AUC for NN + RL


    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_nn, tpr_nn, color='green', lw=2, label=f'Neural Network (AUC = {roc_auc_nn:.2f})')
    plt.plot(fpr_nn_rl, tpr_nn_rl, color='purple', lw=2, label=f'Neural Network + RL (AUC = {roc_auc_nn_rl:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier (AUC = 0.50)')

    # Add labels and legend
    plt.title('ROC Curve: Neural Network vs Neural Network + RL', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the plot
    if(1):
        plt.savefig('out/plots/accuarcy.png')
        print("ROC Curve Image Saved")


def plot_fuel_consumption_vs_complexity():
    # Simulated data: environmental complexity levels and corresponding fuel consumption values
    environmental_complexity = np.arange(1, 11)  # Levels of complexity (1 to 10)
    fuel_consumption_nn = 100 - (environmental_complexity * 3)  # NN model fuel consumption (decreasing trend)
    fuel_consumption_nn_rl = 95 - (environmental_complexity * 4)  # NN + RL model (more efficient)

    # Create the output directory if it doesn't exist
    os.makedirs('out/plots', exist_ok=True)

    # Plotting the graph
    plt.figure(figsize=(8, 6))
    plt.plot(environmental_complexity, fuel_consumption_nn, color='green', marker='o', label='Neural Network (NN)')
    plt.plot(environmental_complexity, fuel_consumption_nn_rl, color='purple', marker='o', label='NN + RL')

    # Graph customization
    plt.title('Fuel Consumption vs. Environmental Complexity', fontsize=14)
    plt.xlabel('Environmental Complexity Level', fontsize=12)
    plt.ylabel('Fuel Consumption (liters/hour)', fontsize=12)
    plt.xticks(ticks=environmental_complexity)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save the graph as an image
    if(1):
     plt.savefig('out/plots/complexity.png')
     print("Complexity image saved")




            
