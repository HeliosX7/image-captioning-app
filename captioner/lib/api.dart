import 'dart:io';
import 'package:dio/dio.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

String uploadUrl = "http://192.168.43.232:5000/api";
String downloadUrl = "http://192.168.43.232:5000/result";

Future getData(String url) async {
  http.Response response = await http.get(url);
  return jsonDecode(response.body);
}

uploadImage(File imageFile,String url) async
{ 
    String base64Image=base64Encode(imageFile.readAsBytesSync());
    Response response= await Dio().post(url,data:base64Image);
    print(response);
}