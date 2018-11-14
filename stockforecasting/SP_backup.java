/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package stockforecasting;

public class SP_backup {
    
    int total_array = 2000;
    int target_array = 1000;
    int test_array_size = 1000;
    int array_start = 25;
    int window_size = 5;
        
    public SP_backup(){
        
        DataFormat df = new DataFormat();
        MovingAverages ma = new MovingAverages();
        FischerTransform ft = new FischerTransform();
        RSI r = new RSI();
               
        double[] movement = new double[target_array];
        double[][] input = new double[total_array][1];
        double[][] target = new double[target_array][1];
        
        
        
        //Prepare for receiving file input
//        String csvFile = "C:\\Users\\Andrew\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\S&P500 Index Data Set2.csv";
        String csvFile = "A:\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\S&P500 Index Data Set2.csv";

        double[] total_prices = df.fileToArray(csvFile, total_array);
//        double[] prices2 = df.fileToArray(csvFile2, l);
//        double[] prices3 = df.fileToArray(csvFile3, l);
//        
//        //Channge recession data from weekly to daily 
//        double[] temp = new double[prices3.length*5];
//        int count = 0;
//        for (int i=0;i<prices3.length;i++){
//            if (i % 5 == 0)
//                count++;
//            temp[i] = prices3[count];
//        }
//        prices3 = temp;
        
        double[] prices = new double[target_array];
        double[] test_prices = new double[test_array_size];
        
        System.arraycopy(total_prices, 0, prices, 0, target_array);
        System.arraycopy(total_prices, target_array, test_prices, 0, test_array_size);
        
        double[] fisher = ft.convert(prices);
        double[] rsi = r.convert(prices);
        
//        for (int i=0; i<prices.length; i++){
//            prices[i] /= 1000;
//        }
//        
//        for (int i=0; i<rsi.length; i++){
//            rsi[i] /= 100;
//        }

        double[] SP_avg = ma.SMA(fisher, window_size);
        double[] SP_avg2 = ma.EMA(prices,window_size);
        double[] SP_avg3 = ma.TMA(prices,window_size);
        double[] SP_avg4 = ma.BMA(prices,window_size);
//        double[] NSDQ_avg = ma.SMA(prices2,n);
//        double[] Recession = ma.SMA(prices3,n);

        double[] SP_trend = df.transform(prices,window_size);
//        double[] NSDQ_trend = df.transform(prices2,n);
//        double[] Recession_trend = df.transform(prices3,n);
        
        
        input = df.timeSeries(SP_avg,20);
        //double[][] input2 = df.timeSeries(SP_avg,10);
        //input = df.concat(input1, input2);
        //input = df.merge(fisher, rsi);
        //, Recession, Recession_trend

        for (int i=1;i<SP_avg.length;i++){
            movement[i-1] = df.checkMovement(SP_avg[i-1], SP_avg[i]);  
            //System.out.println(SP_avg[i-20] + " - " + movement[i-20] + " - " + SP_avg[i]);
        }
        movement[SP_avg.length-1] = 1;

        target = df.make2D(movement);

        input = df.cropArray(input,array_start,input.length);
        target = df.cropArray(target,array_start,target.length);

        ANN net = new ANN();
        net.setHiddenNeurons(40);
        //net.setHiddenNeurons2(15);
        net.setErr(0.1);
        net.setLrc(0.7);
        net.setMomentum(0.5);
        net.modifyValues(true, 0.01);
        net.details(false);
        net.printInputs(input, target);
        net.train(input, target);    
        
        
        //***** Testing Simulation
        
        //Prepare data
        double test_result = 0;
        double[] test_fisher = ft.convert(test_prices);
        double[] test_SMA = ma.SMA(test_fisher, window_size);
        double[][] test_values = df.timeSeries(test_SMA, 20);
        double[] test_movement = new double[test_array_size];
        for (int i=1;i<test_SMA.length;i++){
            test_movement[i-1] = df.checkMovement(test_SMA[i-1], test_SMA[i]);  
        }
        movement[test_SMA.length-1] = 1;
        target = df.make2D(test_movement);
        
        //Run Simulation
        for (int i=window_size; i<test_values.length; i++){
            test_result += net.test(test_values[i], target[i]);
        }
        
        //Print results
        double accuracy = ((test_array_size-test_result) / test_array_size) * 100;
        System.out.println("Accuracy: "+accuracy+"%");
    }   
    
}
