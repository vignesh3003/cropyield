import React, { useState } from "react";
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  Button,
  ScrollView,
  Alert,
} from "react-native";
import api from "./api";

export default function App() {
  const [crop, setCrop] = useState("Rice");
  const [year, setYear] = useState("2020");
  const [season, setSeason] = useState("Kharif");
  const [stateName, setStateName] = useState("Karnataka");
  const [area, setArea] = useState("100");
  const [rainfall, setRainfall] = useState("900");
  const [fertilizer, setFertilizer] = useState("50");
  const [pesticide, setPesticide] = useState("2");
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    setPrediction(null);
    const payload = {
      Crop: crop,
      Crop_Year: Number(year),
      Season: season,
      State: stateName,
      Area: Number(area),
      Annual_Rainfall: Number(rainfall),
      Fertilizer: Number(fertilizer),
      Pesticide: Number(pesticide),
    };

    try {
      const res = await api.postPredict(payload);
      setPrediction(res.prediction);
    } catch (err) {
      const msg =
        err && err.errors
          ? JSON.stringify(err.errors)
          : (err && err.error) || JSON.stringify(err);
      Alert.alert("Prediction error", msg.toString());
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Crop Yield Predictor</Text>

      <TextInput
        style={styles.input}
        value={crop}
        onChangeText={setCrop}
        placeholder="Crop (e.g. Rice)"
      />
      <TextInput
        style={styles.input}
        value={year}
        onChangeText={setYear}
        placeholder="Year"
        keyboardType="numeric"
      />
      <TextInput
        style={styles.input}
        value={season}
        onChangeText={setSeason}
        placeholder="Season (Kharif/Rabi)"
      />
      <TextInput
        style={styles.input}
        value={stateName}
        onChangeText={setStateName}
        placeholder="State"
      />
      <TextInput
        style={styles.input}
        value={area}
        onChangeText={setArea}
        placeholder="Area (ha)"
        keyboardType="numeric"
      />
      <TextInput
        style={styles.input}
        value={rainfall}
        onChangeText={setRainfall}
        placeholder="Annual Rainfall (mm)"
        keyboardType="numeric"
      />
      <TextInput
        style={styles.input}
        value={fertilizer}
        onChangeText={setFertilizer}
        placeholder="Fertilizer"
        keyboardType="numeric"
      />
      <TextInput
        style={styles.input}
        value={pesticide}
        onChangeText={setPesticide}
        placeholder="Pesticide"
        keyboardType="numeric"
      />

      <View style={styles.button}>
        <Button
          title={loading ? "Predicting…" : "Predict"}
          onPress={handlePredict}
          disabled={loading}
        />
      </View>

      {prediction !== null && (
        <View style={styles.result}>
          <Text style={styles.resultText}>Predicted yield: {prediction}</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    paddingTop: 60,
    backgroundColor: "#fff",
    minHeight: "100%",
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    marginBottom: 20,
    textAlign: "center",
  },
  input: {
    borderWidth: 1,
    borderColor: "#ddd",
    padding: 10,
    borderRadius: 8,
    marginBottom: 12,
  },
  button: { marginTop: 6, marginBottom: 12 },
  result: {
    backgroundColor: "#f0f8ff",
    padding: 12,
    borderRadius: 8,
    marginTop: 10,
  },
  resultText: { fontSize: 18, fontWeight: "600" },
});
