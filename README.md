# db-systems-typesql

[TypeSQL](https://arxiv.org/abs/1804.09769) is a Knowledge-based Type-Aware Neural sequence-to-sequence model that attempts to convert natural language sentences into SQL queries. <br>
In this Repository, it is analyzed the architecture, the structure of the model and their performance in comparison with its main competitors [SQLNet](https://arxiv.org/abs/1711.04436) and [Seq2SQL](https://arxiv.org/abs/1709.00103).

***

### Before starting:

Some technical differences between this Code and the author's implementation. 
<br>
<table style="margin-right: auto; width:85%;font-size:15px;border:1px;">
  <tr style="font-size:115%;font-weight:bold;font-style:italic; ">
    <th > </th> <th >Authos</th> <th>ThisProject</th>     
  </tr>
  <tr>
    <td>Epochs</td> <td>100</td> <td>7</td>  
  </tr>
  <tr>
    <td>Python</td> <td>py2.7 </td> <td>py3.8</td> 
  </tr>
    <tr> 
    <td>Glove</td> <td>  42B.300d </td>  <td> 6B.300d</td> 
  </tr> 
</table>
<B><i>For the experiments it was used Google Colab's GPU</i></B>

<br> <br> 


## Run the Model
In order to run the model.
* Data: Download data from this [link](https://drive.google.com/file/d/1CGIRCjwf2bgmWl3UyjY1yJpP4nU---Q0/view)
* Prerequisites: Install ` records==0.5.2 `and download Embeddings (Glove version of your will and [paraphrase embedding](https://drive.google.com/file/d/1iWTowxEG1-KZyq-fHP6cb6dNqMh4eHiN/view) )

* Training: For training:
     * Type `python MyTypeSQL.py` to take a description of the model.
     * There are offered a set of training options like: ` Optimizer=(Adam, Adadelta, SGD), UnitNumbers=(60, 120,180), Content=(0,1) etc.`
     * Customize your model according to your own needs.
* Testing: To run the default model (Epochs:7, Units:180, Opt:Adam) type `python Testing.py`  or follow the description to test your model.
* Run competitors with **py2Competitors**
 
<br>


<B><i>Sources:*</i></B>
TypeSQL: https://github.com/taoyds/typesql <br>
SQLNet:  https://github.com/xiaojunxu/SQLNet <br>
Seq2SQL: https://github.com/shanelleroman/seq2sql

WikiSQL: https://github.com/salesforce/WikiSQL 

***
 
***

## Current Trend  in Text-to-Sql Query

Despite the significant improvement (especially on WHERE Clauses) that typesql offers, if we look at the leaderboard of [WikiSQL](https://github.com/salesforce/WikiSQL) we will find out that TypeSql is outperformed by other models relatively easy. 
Currently it posses the 17th position among the 25 best models for WikiSQL.

<br>

<table style="margin-right: auto; width:85%;font-size:15px;border:1px;">
  <tr style="font-size:115%;font-weight:bold;font-style:italic; ">
    <th >Rank </th> <th >Paper</th>  
  </tr>
  <tr>
 <td>1</td> 
    <td>IE-SQL
+Execution-Guided Decoding
(Ma 2020)
(Ping An Life, AI Team)</td>  
  </tr>
  <tr>
 <td>2</td> 
    <td>HydraNet
+Execution-Guided Decoding
(Lyu 2020)
(Microsoft Dynamics 365 AI)
  </tr>
  <tr>
 <td>3</td> 
    <td>
 X-SQL
+Execution-Guided Decoding
(He 2019)</td> 
  </tr>
  <tr>
 <td>4</td> 
    <td>BRIDGE^
+Execution-Guided Decoding
(anonymous)  <td>  
  </tr>
  <tr>
 <td>5</td> 
    <td>(Guo 2019)+Execution-Guided Decoding with BERT-Base-Uncased  </td>  
  </tr>
 <tr>
 <td>...</td> <td>...</td>  
</tr>
  <tr>
    <td>16</td> 
    <td>TypeSQL (Yu 2018)  </td>  
  </tr>
    <tr>
    <td>17</td> 
    <td>PT-MAML (Huang 2018)  </td>  
  </tr>
   <tr>
    <td>18</td> 
    <td>(Guo 2018) </td>  
  </tr>
   <tr>
    <td>19</td> 
    <td>SQLNet (Xu 2017)  </td>  
  </tr>
      <td>21</td> 
    <td>Seq2SQL (Zhong 2017)  </td>  
  </tr
</table>



