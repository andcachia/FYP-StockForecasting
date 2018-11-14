
package stockforecasting;

import java.awt.BorderLayout;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextArea;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class StockForecasting extends JFrame{

    Random rand = new Random();
      
    String[][] s1 = new String[10][2];
    String[][] s2 = new String[6][2];
    String[] temp;
    
    static int counter = 0;    
    double[] input = new double[5];
    double[] hidden = new double[4];
    double[] OUT = new double[3];
    double[] target = new double[3];
    double[] error = new double[3];
    double[] delta;
    double[][] WH;
    double[][] WO;
    double[][] WH_delta;
    double[][] WO_delta;
    
    int input_length = 4;
    int hidden_length = 3;
    int output_length = 2;
    
    final double lrc = 0.2;
    final double err = 0.2;
    final double momentum = 0.9;
    
    double newDelta = 0;
    int badEpoch = 1;
    boolean badFact = false;
    
    boolean details = false;
    double PosX = 0;
    double PosY = 0;
    
    XYSeries series1;
    XYSeriesCollection dataset;
    JFreeChart chart;
        
    public StockForecasting(){
        //Frame settings
        setTitle("S&P 500");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());
        setSize(1200, 650);
        setVisible(true);
        
        //Chart settings and variables
        series1 = new XYSeries("First");
        dataset = new XYSeriesCollection();
        dataset.addSeries(series1);
        chart = createChart(dataset);
        JPanel content = new JPanel(new BorderLayout());
        ChartPanel chartPanel = new ChartPanel(chart);
        content.add(chartPanel);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(content);
        
        //makes frame paint immediately
        validate();
        
        //Prepare for receiving file input
        //String csvFile = "A:\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\S&P500 Index Data Set2.csv";
        String csvFile = "A:\\Dropbox\\FYP - Andrew Cachia\\Data Sets\\S&P500 Index Data Set2.csv";
	BufferedReader br = null;
	String line = "";
	String cvsSplitBy = ",";
        
        //randomize weights
        WH = randomizeWeights(input_length,hidden_length,-1,1);
        WO = randomizeWeights(hidden_length,output_length,-1,1);
        //weights to hold delta values used for momentum in back propogation
        //These are initiall set to 0
        WH_delta = randomizeWeights(input_length,hidden_length,0,0);
        WO_delta = randomizeWeights(hidden_length,output_length,0,0);
        
        
        //Receive values from file and store them in a 2d array, with each row contianing input and target
        try {
		br = new BufferedReader(new FileReader(csvFile));
                int count = 0;
		while ((line = br.readLine()) != null && (count < s1.length)) {
		        // use comma as separator to split between input and target
			s1[count] = line.split(cvsSplitBy);
                        count++;
                }
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
       
        
        //***************************************************

        //shuffleArray(s1);
        
        /*for (int i=0;i<s1.length;i++){
            s1[i] = s[i];
        }
        for (int i=0;i<6;i++){
            s2[i] = s[i+26];
        }*/
        
        
               
        //*****************************Training**********************************
        while (badEpoch != 0){
            if (details) System.out.println("\n******  Epoch Number: " + counter + " *********");
            badEpoch = 0;
            counter++;
            for (int count=4;count<s1.length - 1;count++){
                //placing the input and target vector bits into seperate "neurons"
                input = populateNeurons(new String[]{ s1[count][1], s1[count-1][1], s1[count-2][1], s1[count-3][1]});
                target = checkMovement(s1[count][1], s1[count+1][1]);
                badFact = false;
                delta = new double[2];
                
                //program carries out first forward pass
                hidden = forwardPass(input, WH);
                OUT = forwardPass(hidden, WO);
                
                //computes error value and delta for each output neuron
                if (details) System.out.println("Vector: " + count);
                for (int i=0;i<output_length;i++){
                    error[i] = (target[i] - OUT[i]);
                    delta[i] = OUT[i]*(1-OUT[i])*error[i];
                    if (Math.abs(error[i]) > err) badFact = true;
                    if (details = true){
                        System.out.print("\tOUT: " + OUT[i] + "\t ");
                        System.out.print("\tTarget: " + target[i] + "\t ");                    
                        System.out.print("\tError: " + error[i] + "\t ");                    
                        System.out.println("\tDelta: " + delta[i]);
                    }
                }

                if (badFact){
                    badEpoch++;
                    //passes delta values and performs first backward pass
                        backwardPass(WO, WO_delta, hidden, delta);
                    //computes new delta values for hidden layer, using old delta values
                        delta = newDelta(delta, WO, hidden);
                    //performs second backward pass using new delta values
                        backwardPass(WH, WH_delta, input, delta);
                }
            }
            if (details){
                System.out.println("Layer 1 weights: ");
                printWeights(WH);
                System.out.println("Layer 2 weights: ");
                printWeights(WO);
            }
            updateGraph();
        }
        
        
        //*******************Testing****************
        /*JFrame frame = new JFrame("Testing Results");
        frame.setSize(800, 300);
        frame.setVisible(true);
        JTextArea textArea = new JTextArea();
        frame.add(textArea);
        for (int i=0; i<s2.length;i++){           
            
            textArea.append("Input: " + s2[i][0]);
            
            //almost same process as before, however only forward pass is done (no backward pass)
            input = populateNeurons(s2[i][0]);
            target = populateNeurons(s2[i][1]);
            hidden = forwardPass(input,WH);
            OUT = forwardPass(hidden, WO);
            
            textArea.append("\tExpected: " + s2[i][1]);
            
            textArea.append("\tActual:  ");
            for(double d : OUT){
                textArea.append(d + "   ");
            }
            textArea.append("\n");
        }*/
    }
    
    
    
    //********************************* functions ***************************************
    public double[] populateNeurons(String[] s){
        int count = 0;
        double[] d = new double[s.length];
        while (count<s.length){
            d[count] = Double.parseDouble(s[count]);
            count++;
        }
        return d;
    }
    
    public double[] checkMovement(String currentPrice, String newPrice){
        double current_price = Double.parseDouble(currentPrice);
        double new_price = Double.parseDouble(newPrice);
        if (new_price > current_price) return new double[]{ 1 , 0 };
        else if (new_price < current_price) return new double[]{ 0 , 1 };
        else return new double[]{ 0 , 0 };
    }
    
    // Implementing Fisher-Yates shuffle
    public void shuffleArray(String[][] ar)
    {
      Random rnd = new Random();
      for (int i = ar.length - 1; i > 0; i--)
      {
        int index = rnd.nextInt(i + 1);
        // Simple swap
        String a0 = ar[index][0];
        String a1 = ar[index][1];
        ar[index][0] = ar[i][0];
        ar[index][1] = ar[i][1];
        ar[i][0] = a0;
        ar[i][1] = a1;
      }
    }
    
    public double[][] randomizeWeights(int i, int j, double min, double max){
        double[][] c = new double[i][j];
        for (int cnt1 = 0; cnt1<i; cnt1++){
            for (int cnt2 = 0; cnt2<j; cnt2++){
                c[cnt1][cnt2] = min + (rand.nextDouble()*(max - min));
            }
        }
        return c;
    }

    
    public double[] forwardPass(double[] vector, double[][] WH){
        double[] NET = new double[vector.length-1];
        for(int i=0;i<vector.length-1;i++){
            double x=0;
            for(int j=0;j<vector.length;j++){
                x += (vector[j] * WH[j][i]);
            }
            NET[i] = x;
        }
        
        for(int n=0;n<NET.length;n++){
            NET[n] = sigmoid(NET[n]);
        }
        
        return NET;
    }
    
    public void backwardPass(double[][] w, double[][] w_delta, double[] layer_in, double[] delta){
        double temp;
        for (int i=0; i<layer_in.length; i++){
            for(int j=0;j<layer_in.length-1;j++){
                temp = (lrc * layer_in[i] * delta[j]) + (momentum * w_delta[i][j]);
                w[i][j] = w[i][j] + temp;
                w_delta[i][j] = temp;
            }
        }
    }
    
    public double[] newDelta(double[] delta, double[][] WO, double[] hidden){
        int dlength = delta.length+1;
        double[] d = new double[dlength];
        for (int i=0;i<dlength;i++){            
            for (int j=0;j<delta.length;j++){
                d[i] += delta[j] * WO[i][j];
            }
            d[i] = d[i] * (hidden[i] * (1 - hidden[i]));
        }
        return d;
    }
    
    public double sigmoid(double d){
        return (1 / (1 + Math.exp(-d)));
    }
    
    public void printWeights(double[][] w){
        for (int i=0;i<w.length;i++){
            for(int j=0;j<w[0].length;j++){
                System.out.print(w[i][j] + " ");
            }
            System.out.println("");
        }
    }
    
    
    public void updateGraph(){      
            PosX++;
            PosY = badEpoch;
            series1.add(PosX, PosY);
    }
    
     private JFreeChart createChart(final XYDataset dataset) {
        
        // create the chart...
        final JFreeChart chart = ChartFactory.createXYLineChart(
            "Bad facts over time",      // chart title
            "Number of Epochs",                      // x axis label
            "Bad Facts",                      // y axis label
            dataset,                  // data
            PlotOrientation.VERTICAL,
            true,                     // include legend
            true,                     // tooltips
            false                     // urls
        );
        
        return chart;
     }
    
     public static void main(String[] args) {
        new StockForecasting();        
    }
    
}
