/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package stockforecasting;

public class SP_NSDQ {
    
    int total_array = 2000;
    int training_array_size = 1000;
    int test_array_size = total_array - training_array_size;
    int array_start = 25;
    int window_size = 5;
        
    public SP_NSDQ(){
        
        DataFormat df = new DataFormat();
        MovingAverages ma = new MovingAverages();
               
        double[] movement = new double[total_array];
        double[][] input;
        double[][] target;
        
        //Prepare for receiving file input
//        String csvFile = "C:\\Users\\Andrew\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\S&P500 Index Data Set2.csv";
//        String csvFile2 = "C:\\Users\\Andrew\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\NASDAQb.csv";
//        String csvFile3 = "C:\\Users\\Andrew\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\FTSEb.csv";
        String csvFile = "A:\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\S&P500 Index Data Set2.csv";
        String csvFile2 = "A:\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\NASDAQb.csv";
        String csvFile3 = "A:\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\Google Trends\\recession2.csv";

        double[] total_prices_SP= df.fileToArray(csvFile, total_array);
        double[] total_prices_NSDQ = df.fileToArray(csvFile2, total_array);
        double[] total_values_recession = df.fileToArray(csvFile3, total_array/5);
        
        //Channge recession data from weekly to daily
        double[] temp = new double[total_values_recession.length*5];
        int count = 0;
        for (int i=0;i<total_values_recession.length;i++){
            if (i % 5 == 0)
                count++;
            temp[i] = total_values_recession[count];
        }
        total_values_recession = temp;
       
        
//        for (int i=0; i<prices.length; i++){
//            training_prices[i] /= 1000;
//        }
//        
//        for (int i=0; i<rsi.length; i++){
//            rsi[i] /= 100;
//        }

        double[] SP_avg = ma.SMA(total_prices_SP, window_size);
        double[] NSDQ_avg = ma.SMA(total_prices_NSDQ,window_size);
        double[] Recession = ma.SMA(total_values_recession,window_size);

        double[] SP_trend = df.transform(total_prices_SP,window_size);
        double[] NSDQ_trend = df.transform(total_prices_NSDQ,window_size);
        double[] Recession_trend = df.transform(total_values_recession,window_size);
        
        for (int i=1;i<SP_avg.length;i++){
            movement[i-1] = df.checkMovement(SP_avg[i-1], SP_avg[i]);  
        }
        movement[SP_avg.length-1] = 1;

        input = df.merge(SP_avg, SP_trend,NSDQ_avg,NSDQ_trend,Recession,Recession_trend);
        target = df.make2D(movement);

        double[][] training_input = df.cropArray(input,array_start,training_array_size);
        double[][] training_target = df.cropArray(target,array_start,training_array_size);

        ANN net = new ANN();
        net.setHiddenNeurons(20);
        net.setErr(0.3);
        net.setLrc(0.4);
        net.setMomentum(0.9);
        net.modifyValues(false, 0.01);
        net.details(false);
        net.setConvergenceLimit(375);
        net.printInputs(training_input, training_target);
        net.train(training_input, training_target);    
        
        
        //***** Testing Simulation
        
        //Prepare data
        double[][] testing_input = df.cropArray(input,training_array_size,total_array);
        double[][] testing_target = df.cropArray(target,training_array_size,total_array);
        double test_result = 0;
        
        //Run Simulation
        for (int i=window_size; i<testing_input.length; i++){
            test_result += net.test(testing_input[i], testing_target[i]);
        }
        
        //Print results
        double accuracy = ((test_array_size-test_result) / test_array_size) * 100;
        System.out.println("Accuracy: "+accuracy+"%");
    }   
    
}
