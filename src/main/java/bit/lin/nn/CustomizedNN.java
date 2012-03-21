package bit.lin.nn;

import java.util.Date;

import org.joone.engine.Monitor;
import org.joone.engine.NeuralNetEvent;
import org.joone.engine.NeuralNetListener;
import org.joone.net.NeuralNet;

import bit.lin.utils.JUtils;

public class CustomizedNN implements NeuralNetListener {
	protected NeuralNet _nn;
	protected int _patterns = 723412 * 20;
	protected int _cicles = 1;
	protected double _learningRate = 0.8;
	protected double _momentum = 0.3;
	int i = 1;
	private JUtils _jUtils = new JUtils();

	public void setNeuralnet(NeuralNet _sortnet) {
		this._nn = _sortnet;
	}

	public NeuralNet getNeuralNet() {
		return _nn;
	}

	public void setLearner(String className) {
		_nn.getMonitor().addLearner(i, className);
		_nn.getMonitor().setLearningMode(i);
		i++;
	}

	public int getCicles() {
		return _cicles;
	}

	public void setCicles(int cicles) {
		this._cicles = cicles;
		_nn.getMonitor().setTotCicles(_cicles);
	}

	public double getMomentum() {
		return _momentum;
	}

	public void setMomentum(double _momentum) {
		this._momentum = _momentum;
		_nn.getMonitor().setMomentum(_momentum);
	}

	public double getLearningRate() {
		return _learningRate;
	}

	public void setLearningRate(double learningRate) {
		this._learningRate = learningRate;
		_nn.getMonitor().setLearningRate(_learningRate);
	}

	public int getPatternCount() {
		return _patterns;
	}

	public void setPatternCount(int patternCount) {
		this._patterns = patternCount;
		_nn.getMonitor().setTrainingPatterns(_patterns);
	}

	public void cicleTerminated(NeuralNetEvent arg0) {
		Monitor m = (Monitor) arg0.getSource();
		int i = m.getCurrentCicle();
		System.out.format("Epoch %d remaining - RMSE = %f.\n", i,
				m.getGlobalError());
	}

	double _prev = -10;

	public void errorChanged(NeuralNetEvent arg0) {
		Monitor m = (Monitor) arg0.getSource();
		if (m.getCurrentCicle() == 1)
			_prev = m.getGlobalError();
		else {
			if (Math.abs(_prev - m.getGlobalError()) < 0.00002)
				_nn.stop();
			_prev = m.getGlobalError();
		}

	}

	public void netStarted(NeuralNetEvent arg0) {
		System.out.println("Training started at "
				+ new Date(System.currentTimeMillis()));
	}

	public void netStopped(NeuralNetEvent arg0) {
		System.out.println("Training finished at "
				+ new Date(System.currentTimeMillis()));
	}

	public void netStoppedError(NeuralNetEvent arg0, String arg1) {
	}

	public void train(boolean isListening) {
		if (isListening)
			_nn.getMonitor().addNeuralNetListener(this);
		_nn.go();
	}

	public void saveNeuralNet(String fName) {
		_jUtils.saveNeuralNet(_nn, fName);
		System.out.println("Neural Net has been serialized!");
	}
	
}
