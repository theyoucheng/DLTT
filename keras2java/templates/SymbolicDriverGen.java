import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
 
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import java.util.ArrayList;
 
public class SymbolicDriverGen
{
    
    public static void convert(String jsonFile, String javaFile, String dataFile, String failFile)
    {
        String prog = "";
        prog += "package dnn;\n";
        prog += "\n";
        prog += "import java.io.BufferedReader;\n";
        prog += "import java.io.File;\n";
        prog += "import java.io.FileInputStream;\n";
        prog += "import java.io.FileNotFoundException;\n";
        prog += "import java.io.FileReader;\n";
        prog += "import java.io.IOException;\n";
        prog += "\n";
        prog += "import gov.nasa.jpf.symbc.Debug;\n";
        prog += "import gov.nasa.jpf.symbc.DNN;\n";

        prog += "\n";
        prog += "\n";
        prog += "public class SymbolicDriver {\n";
        prog += "\n";
        prog += " static final int IMG_HEIGHT = 28;\n";
        prog += " static final int IMG_WIDTH = 28;\n";
        prog += "\n";
        prog += "\n";




        JSONParser jsonParser = new JSONParser();
         
        try (FileReader reader = new FileReader(jsonFile))
        {
            //Read JSON file
            Object obj = jsonParser.parse(reader);
 
            JSONObject model = (JSONObject) obj;
            //System.out.println(model);
            Integer nlayers = (int)(long)model.get("nlayers");
            System.out.println("### Number of layers: " + nlayers);
            JSONArray j_union = (JSONArray)model.get("union");
            ArrayList<Integer> union = parseJArray(j_union);
            System.out.println("### Union indices: " + union);
            prog += " public static void main(String[] args) throws Exception {\n";
            prog += "   /* Read internal data */\n";
            prog += "   InternalData internalData = new InternalData();\n";
            prog += "   DNN.readDataFromFiles(\"" + dataFile + "\");\n";
            for (int e : union) {
              prog += "   internalData.biases" + e + " = DNN.getBiases" + e + "();\n";
            }
            for (int e : union) {
              prog += "   internalData.weights" + e + " = DNN.getWeights" + e + "();\n";
            }

            prog += "   // Reading input from file\n";
            prog += "   System.out.println(\"FAILING TESTS\");\n";
            prog += "   String FailFile = \"" + failFile + "\";\n";
            prog += "   File file = new File(FailFile);\n";
            prog += "   BufferedReader br = new BufferedReader(new FileReader(file));\n";
            prog += "   double[][][] a = new double[IMG_HEIGHT][IMG_WIDTH][1];\n";
            prog += "   String st;\n";
            prog += "   while ((st = br.readLine()) != null) {\n";
            prog += "     System.out.println(\"INPUT:\" + st);\n";
            prog += "     String[] values = st.split(\",\");\n";
            prog += "     int index = 0;\n";
            prog += "     while (index < values.length) {\n";
            prog += "       for (int i = 0; i <  IMG_HEIGHT; i++)\n";
            prog += "         for (int j = 0; j < IMG_WIDTH; j++)\n";
            prog += "           for (int k = 0; k < 1; k++)\n";
            prog += "           {\n";
            prog += "             Double val = Double.valueOf(values[index]);\n";
            prog += "             index++;\n";
            prog += "             a[i][j][k] = (double)(val/255.0);\n";
            prog += "           }\n";
            prog += "     }\n";
            prog += "     System.out.println(\"outcome \"+run(internalData, a));\n";
            prog += "     System.out.println(Debug.PC4Z3());\n";
            prog += "     break;\n";
            prog += "   }\n";
            prog += "   br.close();\n";
            prog += " }\n";

            // run method: BEGIN
            prog += " public static int run(InternalData internal, double[][][] input) {\n";
            ArrayList<Integer> input_shape = parseJArray((JSONArray)(getJObj(0, model).get("inp_sp")));
            System.out.println("input shape: " + input_shape);
            boolean fc = false;
            int channel = -1, row = -1, column = -1;
            if (input_shape.size() > 2) {
              channel = input_shape.get(3);
              row = input_shape.get(1);
              column = input_shape.get(2);
            } else { fc = true; }

            for (int l=0; l<nlayers; l++) {
              JSONObject layer = getJObj(l, model);
              String name = (String)layer.get("name");
              System.out.println("### name: " + name);
              ArrayList<Integer> _inp_sp = parseJArray((JSONArray)layer.get("inp_sp"));
              ArrayList<Integer> _out_sp = parseJArray((JSONArray)layer.get("out_sp"));
              boolean is_conv = (boolean)layer.get("is_conv");
              boolean is_padding = false;
              if (is_conv) is_padding = (boolean)layer.get("is_padding");
              boolean is_dense = (boolean)layer.get("is_dense");
              boolean is_activation = (boolean)layer.get("is_activation");
              boolean is_maxpooling = (boolean)layer.get("is_maxpooling");
              boolean is_flatten = (boolean)layer.get("is_flatten");
              boolean is_relu = (boolean)layer.get("is_relu");
              prog += "   // layer " + l + ": " + name + "\n";
              ArrayList<Integer> sp = null;
              ArrayList<Integer> bsp = null;
              if (is_conv || is_dense) {
                 sp = parseJArray((JSONArray)layer.get("w_sp"));
                 bsp = parseJArray((JSONArray)layer.get("b_sp"));
              }
              if (is_conv) {
                prog += "   double[][][] layer" + l + "=new double[" + _out_sp.get(1) + "][" + _out_sp.get(2) + "][" + _out_sp.get(3) + "];\n";
                ArrayList<Integer> kernel_size = parseJArray((JSONArray)layer.get("kernel_size"));
                // ASSUMPTION: no padding!
                if (is_padding) {
                
                } else {
                  prog += "   for(int i=0; i<" + _out_sp.get(1) + "; i++)\n";
                  prog += "     for(int j=0; j<" + _out_sp.get(2) + "; j++)\n";
                  prog += "       for(int k=0; k<" + _out_sp.get(3) + "; k++)\n";
                  prog += "       {\n";
                  prog += "         layer" + l + "[i][j][k]=internal.biases" + l + "[k];\n"; 
                  prog += "         for(int I=0; I<" + kernel_size.get(0) + "; I++)\n";
                  prog += "            for(int J=0; J<" + kernel_size.get(1) + "; J++)\n";
                  prog += "              for(int K=0; K<" + sp.get(2) + "; K++)\n";
                  if (l == 0) 
                    prog += "               layer" + l + "[i][j][k]+=internal.weights"+l+"[I][J][K][k]*input[i+I][j+J][K];\n";
                  else
                    prog += "               layer" + l + "[i][j][k]+=internal.weights"+l+"[I][J][K][k]*layer"+(l-1)+"[i+I][j+J][K];\n";
                  if (is_relu)
                    prog += "       if (layer"+l+"[i][j][k]<0) layer"+l+"[i][j][k] = 0; // relu\n";
                  prog += "       }\n";
                }
              }
              else if (is_dense) {
                prog += "   double[] layer"+l+"=new double["+_out_sp.get(1)+"];\n";
                prog += "   for(int i=0; i<"+sp.get(1)+"; i++)\n";
                prog += "   {\n";
                prog += "     layer"+l+"[i]=internal.biases"+l+"[i];\n";
                prog += "     for(int I=0; I<"+sp.get(0)+"; I++)\n";
                if (l==0)
                  prog += "       layer"+l+"[i]+=internal.weights"+l+"[I][i]*input[I];\n";
                else
                  prog += "       layer"+l+"[i]+=internal.weights"+l+"[I][i]*layer"+(l-1)+"[I];\n";

                if (is_relu)
                  prog += "   if (layer"+l+"[i]<0) layer"+l+"[i] = 0; // relu\n";

                prog += "   }\n";
              }
              else if (is_flatten) {
                int s=_inp_sp.get(1) * _inp_sp.get(2) * _inp_sp.get(3);
                prog += "   double[] layer"+l+"=new double["+s+"];\n";
                prog += "   for(int i=0; i<"+s+"; i++)\n";
                prog += "   {\n";
                prog += "     int d0=i/"+_inp_sp.get(2) * _inp_sp.get(3) + ";\n";
                prog += "     int d1=(i%"+_inp_sp.get(2) * _inp_sp.get(3)+")/"+_inp_sp.get(3)+";\n";
                prog += "     int d2=i-d0*"+_inp_sp.get(2)*_inp_sp.get(3)+"-d1*"+_inp_sp.get(3)+";\n";
                prog += "     layer"+l+"[i]=layer"+(l-1)+"[d0][d1][d2];\n";
                prog += "   }\n";
              }
              else if (is_maxpooling) {
                ArrayList<Integer> pool_size = parseJArray((JSONArray)layer.get("pool_size"));
                prog += "    double[][][] layer"+l+"=new double["+_out_sp.get(1)+"]["+_out_sp.get(2)+"]["+_out_sp.get(3)+"];\n";
                prog += "    for(int i=0; i<"+_out_sp.get(1)+"; i++)\n";
                prog += "      for(int j=0; j<"+_out_sp.get(2)+"; j++)\n";
                prog += "        for(int k=0; k<"+_out_sp.get(3)+"; k++)\n";
                prog += "        {\n";
                prog += "          layer"+l+"[i][j][k]=0;\n";
                prog += "          for(int I=i*"+pool_size.get(0)+"; I<(i+1)*"+pool_size.get(0)+"; I++)\n";
                prog += "            for(int J=j*"+pool_size.get(1)+"; J<(j+1)*"+pool_size.get(1)+"; J++)\n";
                prog += "              if(layer"+(l-1)+"[I][J][k]>layer"+l+"[i][j][k]) layer"+l+"[i][j][k]=layer"+(l-1)+"[I][J][k];\n";
                prog += "        }\n";
              }
              else if (is_activation) {
                if (is_relu) {
                  if (_out_sp.size()>2) {
                    prog += "   double[][][] layer"+l+"=new double["+_out_sp.get(1)+"]["+_out_sp.get(2)+"]["+_out_sp.get(3)+"];\n";
                    prog += "   for(int i=0; i<"+_out_sp.get(1)+"; i++)\n";
                    prog += "     for(int j=0; j<"+_out_sp.get(2)+"; j++)\n";
                    prog += "       for(int k=0; k<"+_out_sp.get(3)+"; k++)\n";
                    prog += "          if(layer"+(l-1)+"[i][j][k]>0) layer"+l+"[i][j][k]=layer"+(l-1)+"[i][j][k];\n";
                    prog += "          else layer"+l+"[i][j][k]=0;\n";
                  }
                  else {
                    prog += "   double[] layer"+l+"=new double["+_out_sp.get(1)+"];\n";
                    prog += "   for(int i=0; i<"+_out_sp.get(1)+"; i++)\n";
                    prog += "     if(layer"+(l-1)+"[i]>0) layer"+l+"[i]=layer"+(l-1)+"[i];\n";
                    prog += "     else layer"+l+"[i]=0;\n";
                  }
                }
                else {
                  if (l!=nlayers-1) {
                    System.out.println("Unrecpgnizable DNN structure...layer " + l );
                    return;
                  }
                  else {
                    prog += "   double[] layer"+l+"=new double["+_out_sp.get(1)+"];\n";
                    prog += "   for(int i=0; i<"+_out_sp.get(1)+"; i++)\n";
                    prog += "     layer"+l+"[i]=layer"+(l-1)+"[i]; // alala\n";
                  }
                }
              }
              else {
                 throw new Exception("Unexpecetd layer type: " + name);
              }

              if (l == nlayers-1) {
                prog += "    int ret=0;\n";
                prog += "    double res=-100000;\n";
                prog += "    for(int i=0; i<"+_out_sp.get(1)+";i++)\n";
                prog += "    {\n";
                prog += "      if(layer"+l+"[i]>res)\n";
                prog += "      {\n";
                prog += "        res=layer"+l+"[i];\n";
                prog += "        ret=i;\n";
                prog += "      }\n";
                prog += "    }\n";
                prog += "    return ret;\n";
              }

            }



            prog += " }\n";
            // run method: END

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ParseException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }





        prog += "}\n";



        try {
          BufferedWriter writer = new BufferedWriter(new FileWriter(javaFile));
          writer.write(prog);
          writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    @SuppressWarnings("unchecked")
    public static void main(String[] args) 
    {
      convert ("dnn.json", "SymbolicDriver.java", "C:\\\\Users\\\\dgopinat\\\\eclipse-workspace\\\\jpf-symbc\\\\src\\\\examples\\\\dnn\\\\data", "C:\\\\Users\\\\dgopinat\\\\eclipse-workspace\\\\jpf-symbc\\\\src\\\\examples\\\\dnn\\\\data\\\\Fail10.csv");
        ////JSON parser object to parse read file
        //JSONParser jsonParser = new JSONParser();
        // 
        //try (FileReader reader = new FileReader("dnn.json"))
        //{
        //    //Read JSON file
        //    Object obj = jsonParser.parse(reader);
 
        //    JSONObject layers = (JSONObject) obj;
        //    System.out.println(layers);
        //    // 
        //    System.out.println("Layer 0");
        //    JSONObject jobj = getJObj(0, layers);
        //    boolean is_relu = (boolean)jobj.get("is_relu");
        //    JSONArray j_inp_sp = (JSONArray)jobj.get("inp_sp");
        //    System.out.println(jobj);
        //    System.out.println(is_relu);
        //    System.out.println(j_inp_sp);
        //    ArrayList<Integer> inp_sp = parseJArray(j_inp_sp);
        //    System.out.println(inp_sp);
        //    System.out.println(inp_sp.get(0));
        //    System.out.println(inp_sp.get(1)==28);
        //    JSONArray j_kernel_size = (JSONArray)jobj.get("kernel_size");
        //    ArrayList<Integer> kernel_size = parseJArray(j_kernel_size);
        //    System.out.println(kernel_size);
 
        //} catch (FileNotFoundException e) {
        //    e.printStackTrace();
        //} catch (IOException e) {
        //    e.printStackTrace();
        //} catch (ParseException e) {
        //    e.printStackTrace();
        //}
    }

    public static ArrayList<Integer> parseJArray(JSONArray jarray)
    {
      ArrayList<Integer> sp = new ArrayList<Integer>();
      for (Object e: jarray) {
        try
        {
          Integer i = (int)(long)e;
          sp.add(i);
        } catch (Exception ex)
        {
          sp.add(0);
        }
      }
      return sp;
    }

    private static JSONObject getJObj(int i, JSONObject layers) 
    {
      return ((JSONObject)layers.get(Integer.toString(i)));
    }
 
}
