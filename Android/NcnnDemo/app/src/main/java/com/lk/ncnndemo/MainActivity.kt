package com.lk.ncnndemo

import android.content.res.AssetManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity: AppCompatActivity(){
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        System.loadLibrary("demo")
        startDemo(assets)
    }
    init {

    }

    external fun startDemo(assetManager: AssetManager)


}