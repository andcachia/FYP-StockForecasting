/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package stockforecasting;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author Andrew
 */
public class BinaryTest {
    
    
    public BinaryTest(){
        
        String[] parts;
        double[][] input = new double[32][5];
        double[][] target = new double[32][3];
        DataFormat df = new DataFormat();
        
        
        //Prepare for receiving file input
        String csvFile = "A:\\Dropbox\\Uni Semester 5\\Business Intelligence\\Training Set2.csv";
	BufferedReader br = null;
	String line = "";
	String cvsSplitBy = ",";  
        
        //Receive values from file and store them in a 2d array, with each row contianing input and movement
        try {
		br = new BufferedReader(new FileReader(csvFile));
                int count = 0;
		while ((line = br.readLine()) != null && count<input.length) {
		        // use comma as separator to split between input and movement
			parts = line.split(cvsSplitBy);
                        input[count] = df.stringToArrayDouble(parts[0]);
                        target[count] = df.stringToArrayDouble(parts[1]);
                        
                        
                        count++;
                }
                
                ANN net = new ANN();
                net.setHiddenNeurons(20);
                //net.setHiddenNeurons2(5);
                net.setErr(0.01);
                net.setLrc(0.4);
                net.setMomentum(0.9);
                net.modifyValues(false, 0.01);
                net.details(true);
                net.printInputs(input, target);
                net.train(input, target);
                        
        } catch (Exception e){
                e.printStackTrace();
	} finally {
		if (br != null) {
			try {
				br.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}        
    }
}
