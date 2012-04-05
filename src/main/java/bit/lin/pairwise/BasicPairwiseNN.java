package bit.lin.pairwise;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.RandomAccessFile;

import org.joone.engine.FullSynapse;
import org.joone.engine.LinearLayer;
import org.joone.engine.Matrix;
import org.joone.engine.Monitor;
import org.joone.engine.Pattern;
import org.joone.io.MemoryInputSynapse;
import org.joone.io.MemoryOutputSynapse;
import org.joone.net.NeuralNet;
import org.joone.util.DynamicAnnealing;

import bit.lin.nn.CustomizedNN;
import bit.lin.utils.JUtils;

public class BasicPairwiseNN extends CustomizedNN {
	String trainSrc = "/home/lins/data/learning to rank/10k4j/Fold1/train.txt";

	public BasicPairwiseNN(int hiddenCount) {
		_nn = new NeuralNet();
		// initial layers
		SigmoidLayer input = new SigmoidLayer("layer1 input");
		SigmoidLayer hidden = new SigmoidLayer("layer2 hidden");
		LinearLayer output = new LinearLayer("layer3 output");
		input.setRows(136);
		hidden.setRows(hiddenCount);
		output.setRows(1);
		// initial synapses
		FullSynapse fs4i2h = new FullSynapse();
		FullSynapse fs4h2o = new FullSynapse();

		input.addOutputSynapse(fs4i2h);
		hidden.addInputSynapse(fs4i2h);
		hidden.addOutputSynapse(fs4h2o);
		output.addInputSynapse(fs4h2o);

		// add synapses and layers to neural net
		_nn.addLayer(input, NeuralNet.INPUT_LAYER);
		_nn.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
		_nn.addLayer(output, NeuralNet.OUTPUT_LAYER);

		// configure
		Monitor m = _nn.getMonitor();
		m.setLearningRate(super._learningRate);
		m.setMomentum(_momentum);
		m.setTrainingPatterns(_patterns);
		m.setTotCicles(_cicles);
		m.setLearning(true);
		m.addNeuralNetListener(this);
		m.addNeuralNetListener(new DynamicAnnealing());
		// m.addLearner(0,"org.joone.engine.RpropLearner");
		// m.setLearningMode(0);
	}

	// 不用自带的训练类
	public void train(boolean isListening) {
		if (isListening)
			_nn.getMonitor().addNeuralNetListener(this);

		// 设置memory输入输出synapse
		MemoryInputSynapse input = new MemoryInputSynapse();
		input.setFirstRow(1);
		input.setAdvancedColumnSelector("3-138");
		_nn.getInputLayer().addInputSynapse(input);

		// 在每层加入memory输出从而实现算法。
		MemoryOutputSynapse menOS1 = new MemoryOutputSynapse();
		MemoryOutputSynapse menOS2 = new MemoryOutputSynapse();
		MemoryOutputSynapse menOS3 = new MemoryOutputSynapse();
		_nn.getLayer("layer1 input").addOutputSynapse(menOS1);
		_nn.getLayer("layer2 hidden").addOutputSynapse(menOS2);
		_nn.getLayer("layer3 output").addOutputSynapse(menOS3);

		try {
			RandomAccessFile raf = new RandomAccessFile(trainSrc, "r");
			while (true) {
				/*
				 * first pattern
				 */
				String line = raf.readLine();
				if ("".equals(line) || line == null)
					break;
				double[][] l1 = str2dobl(line);
				double desired1 = l1[0][0];
				input.setInputArray(l1);// 2-137
				_nn.go();
				// 保存数据
				double[] out11 = ((Pattern) menOS1.getAllPatterns().get(0))
						.getArray();
				double[] out12 = ((Pattern) menOS2.getAllPatterns().get(0))
						.getArray();
				double[] out13 = ((Pattern) menOS3.getAllPatterns().get(0))
						.getArray();

				/*
				 * second pattern
				 */
				line = raf.readLine();
				if ("".equals(line) || line == null)
					break;
				double[][] l2 = str2dobl(line);
				double desired2 = l1[0][0];
				input.setInputArray(l1);// 2-137
				_nn.go();
				// 保存数据
				double[] out21 = ((Pattern) menOS1.getAllPatterns().get(0))
						.getArray();
				double[] out22 = ((Pattern) menOS2.getAllPatterns().get(0))
						.getArray();
				double[] out23 = ((Pattern) menOS3.getAllPatterns().get(0))
						.getArray();

				// 梯度下降
				SigmoidLayer opt = (SigmoidLayer) _nn.getLayer("layer3 output");
				SigmoidLayer hdn = (SigmoidLayer) _nn.getLayer("layer2 hidden");
				LinearLayer ipt = (LinearLayer) _nn.getLayer("layer1 input");
				FullSynapse fs32 = opt.getInputSynapse();
				FullSynapse fs21 = hdn.getInputSynapse();

				double myError = 0;
				// = -aDesired + 1 / (1 + Math.exp(-anOutput))
				// / Math.log(10);

				// 输出层
				Matrix b3 = opt.getBias();
				double[] gOut1 = new double[opt.getRows()];
				double[] gOut2 = new double[opt.getRows()];
				for (int i = 0; i < opt.getRows(); i++) {
					gOut1[i] = (myError * (out13[i] * (1 - out13[i]) + opt
							.getFlatSpotConstant()));
					gOut2[i] = (myError * (out23[i] * (1 - out23[i]) + opt
							.getFlatSpotConstant()));
					b3.value[i][0] += _learningRate * (gOut1[i] - gOut2[i]);
				}
				
				// 输出层和隐藏层直接的突出
				Matrix w32 = fs32.getWeights();
				double[] bOut1 = new double[fs32.getInputDimension()];
				double[] bOut2 = new double[fs32.getInputDimension()];
				for (int x = 0; x < fs32.getInputDimension(); ++x) {
					double s1 = 0;
					double s2 = 0;
					for (int y = 0; y < fs32.getOutputDimension(); ++y) {
						s1 += gOut1[y] * w32.value[x][y];
						s2 += gOut2[y] * w32.value[x][y];
					}
					bOut1[x] = s1;// 用于传到下一个layer
					bOut2[x] = s2;
				}
				for(int x = 0; x < fs32.getInputDimension(); x++) {
		            for(int y = 0; y < fs32.getOutputDimension(); y++) {
//		            	w32.value[x][y]+=_learningRate *(-);
		            }
		        }
				
				// 隐藏层
				gOut1 = new double[hdn.getRows()];
				gOut2 = new double[hdn.getRows()];
				Matrix b2 = hdn.getBias();
				for (int i = 0; i < hdn.getRows(); i++) {
					gOut1[i] = (bOut1[i] * (out12[i] * (1 - out12[i]) + opt
							.getFlatSpotConstant()));
					gOut2[i] = (bOut2[i] * (out22[i] * (1 - out22[i]) + opt
							.getFlatSpotConstant()));
					b3.value[i][0] += _learningRate * (gOut1[i] - gOut2[i]);
				}
				
				// 隐藏层和输入层之间的突触
				Matrix w21 = fs21.getWeights();
				bOut1 = new double[fs21.getInputDimension()];
				bOut2 = new double[fs21.getInputDimension()];
				for (int x = 0; x < fs21.getInputDimension(); ++x) {
					double s1 = 0;
					double s2 = 0;
					for (int y = 0; y < fs21.getOutputDimension(); ++y) {
						s1 += gOut1[y] * w21.value[x][y];
						s2 += gOut2[y] * w21.value[x][y];
					}
					bOut1[x] = s1;// 用于传到下一个layer
					bOut2[x] = s2;
				}
				for(int x = 0; x < fs21.getInputDimension(); x++) {
		            for(int y = 0; y < fs21.getOutputDimension(); y++) {
//		            	w21.value[x][y]+=_learningRate *(-);
		            }
		        }
				
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private double[][] str2dobl(String str) {
		double[][] result = new double[1][138];
		int i = 0;
		for (String s : str.split(";")) {
			result[0][i] = new Double(s);
			i++;
		}
		return result;
	}

	public BasicPairwiseNN() {
		_nn = new NeuralNet();
	}

	public static BasicPairwiseNN restoreNeuralNet(String fName) {
		BasicPairwiseNN nn = new BasicPairwiseNN();
		JUtils ju = new JUtils();
		nn.setNeuralnet(ju.restoreNeuralNet(fName));
		System.out.println("Neural Net has been deserialized!");
		return nn;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

	}

}
