package bit.lin.joonetest;

import java.io.File;
import java.util.Vector;

import org.joone.engine.FullSynapse;
import org.joone.engine.Layer;
import org.joone.engine.LinearLayer;
import org.joone.engine.Monitor;
import org.joone.engine.NeuralNetEvent;
import org.joone.engine.NeuralNetListener;
import org.joone.engine.Pattern;
import org.joone.engine.SigmoidLayer;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.FileInputSynapse;
import org.joone.io.MemoryInputSynapse;
import org.joone.io.MemoryOutputSynapse;
import org.joone.net.NeuralNet;

import bit.lin.utils.JUtils;

public class nn4xorGen implements NeuralNetListener {
	private NeuralNet _nn4xor = new NeuralNet();
	JUtils _jUtils = new JUtils();

	public nn4xorGen() {
		// initial layers
		LinearLayer input = new LinearLayer();
		SigmoidLayer hidden = new SigmoidLayer();
		SigmoidLayer output = new SigmoidLayer();
		input.setRows(2);
		hidden.setRows(3);
		output.setRows(1);
		// initial synapses
		FullSynapse fs4i2h = new FullSynapse();
		FullSynapse fs4h2o = new FullSynapse();
		FileInputSynapse inputData = new FileInputSynapse();
		inputData.setAdvancedColumnSelector("1,2");
		inputData.setInputFile(new File("src/test/resources/xor"));
		TeachingSynapse ts = new TeachingSynapse();
		FileInputSynapse desiredData = new FileInputSynapse();
		desiredData.setAdvancedColumnSelector("3");
		desiredData.setInputFile(new File("src/test/resources/xor"));
		ts.setDesired(desiredData);
		// link synapses to layers
		input.addInputSynapse(inputData);
		input.addOutputSynapse(fs4i2h);
		hidden.addInputSynapse(fs4i2h);
		hidden.addOutputSynapse(fs4h2o);
		output.addInputSynapse(fs4h2o);
		output.addOutputSynapse(ts);
		// add synapses and layers to neural net
		_nn4xor.addLayer(input, NeuralNet.INPUT_LAYER);
		_nn4xor.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
		_nn4xor.addLayer(output, NeuralNet.OUTPUT_LAYER);
		_nn4xor.setTeacher(ts);
		// configure
		Monitor m = _nn4xor.getMonitor();
		m.setLearningRate(0.8);
		m.setMomentum(0.3);
		m.setTrainingPatterns(4);
		m.setTotCicles(10);
		m.setLearning(true);
	}

	public void train(boolean isListening) {
		if (isListening)
			_nn4xor.getMonitor().addNeuralNetListener(this);
		_nn4xor.go();
	}

	public NeuralNet getNeuralNet() {
		return _nn4xor;
	}

	public void saveNeuralNet(String fName) {
		_jUtils.saveNeuralNet(_nn4xor, fName);
		System.out.println("Neural Net has been serialized!");
	}

	public void restoreNeuralNet(String fName) {
		_nn4xor = _jUtils.restoreNeuralNet(fName);
		System.out.println("Neural Net has been deserialized!");
	}

	public void cicleTerminated(NeuralNetEvent arg0) {
		Monitor m = (Monitor) arg0.getSource();
		int i = m.getCurrentCicle();
		if (i % 1000 == 0)
			System.out.format("Epoch %d remaining - RMSE = %f.\n", i,
					m.getGlobalError());

	}

	public void errorChanged(NeuralNetEvent arg0) {
		System.out.println("Error changed.");

	}

	public void netStarted(NeuralNetEvent arg0) {
		System.out.println("Training strarted!");
	}

	public void netStopped(NeuralNetEvent arg0) {
		System.out.println("Training finished!");
	}

	public void netStoppedError(NeuralNetEvent arg0, String arg1) {
	}

	public void validate() {
		double[][] inputArray = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
		Vector v = _jUtils.validate(_nn4xor, inputArray, "1-2");

		for (int i = 0; i < v.size(); i++) {
			Pattern p = (Pattern) v.get(i);
			double[] pattern = p.getArray();
			System.out
					.println("Output pattern#" + (i + 1) + " = " + pattern[0]);
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		nn4xorGen nn = new nn4xorGen();
		nn.train(true);
		// nn.restoreNeuralNet("src/main/resources/xor.nn");
		// nn.validate();
		System.out.println(nn.getNeuralNet().getMonitor().getLearner());
	}
}
