package bit.lin.pointwise;

import java.io.File;

import org.joone.engine.FullSynapse;
import org.joone.engine.LinearLayer;
import org.joone.engine.Monitor;
import org.joone.engine.SigmoidLayer;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.FileInputSynapse;
import org.joone.net.NeuralNet;
import org.joone.script.MacroInterface;
import org.joone.util.DynamicAnnealing;

import bit.lin.nn.CustomizedNN;
import bit.lin.pairwise.sortnet.SortNet;
import bit.lin.utils.JUtils;

public class BasicPointwiseNN extends CustomizedNN {
	String trainSrc = "/home/lins/data/learning to rank/10k4j/Fold1/train.txt";

	public BasicPointwiseNN() {
		_nn = new NeuralNet();
	}

	public BasicPointwiseNN(int hiddenCount) {
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
		// input desired
		FileInputSynapse inputData = new FileInputSynapse();
		inputData.setBuffered(false);
		inputData.setAdvancedColumnSelector("3-138");
		inputData.setInputFile(new File(trainSrc));
		FileInputSynapse desiredData = new FileInputSynapse();
		desiredData.setBuffered(false);
		desiredData.setAdvancedColumnSelector("1");
		desiredData.setInputFile(new File(trainSrc));
		// teaching synapse
		TeachingSynapse ts = new TeachingSynapse();
		// ts.setDesired(inputValidating);
		ts.setDesired(desiredData);
		input.addInputSynapse(inputData);
		input.addOutputSynapse(fs4i2h);
		hidden.addInputSynapse(fs4i2h);
		hidden.addOutputSynapse(fs4h2o);
		output.addInputSynapse(fs4h2o);
		output.addOutputSynapse(ts);
		// add synapses and layers to neural net
		_nn.addLayer(input, NeuralNet.INPUT_LAYER);
		_nn.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
		_nn.addLayer(output, NeuralNet.OUTPUT_LAYER);
		_nn.setTeacher(ts);
		// configure
		Monitor m = _nn.getMonitor();
		m.setLearningRate(super._learningRate);
		m.setMomentum(_momentum);
		m.setTrainingPatterns(_patterns);
		m.setTotCicles(_cicles);
		m.setLearning(true);
		m.addNeuralNetListener(this);
		m.addNeuralNetListener(new DynamicAnnealing());
		m.setBatchSize(1);
//		m.addLearner(0,"org.joone.engine.RpropLearner");
//		m.setLearningMode(0);
	}
	
	public static BasicPointwiseNN restoreNeuralNet(String fName) {
		BasicPointwiseNN pnn = new BasicPointwiseNN();
		JUtils ju = new JUtils();
		pnn.setNeuralnet(ju.restoreNeuralNet(fName));
		System.out.println("Neural Net has been deserialized!");
		return pnn;
	}
	
	public static void main(String[] args) {
		BasicPointwiseNN pnn = new BasicPointwiseNN(20);
		pnn.setCicles(2000);
		pnn.setPatternCount(500);
		pnn.setLearningRate(0.1);
		pnn.train(true);
		pnn.saveNeuralNet("src/main/resources/pointwise.nn");
//		System.out.println(System.getenv().get("PATH"));
	}
}
