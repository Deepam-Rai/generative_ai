# generative_ai
Testing and testing different techniques for generative AI.

----
## Markov chains
Trains on the given data and tries to generate similar sentences.  
High-overview: During training creates a matrix that maps last `k` tokens to the immediately following next token. During inference, given sequence of tokens, refers to the map-matrix and predicts next token.  
### Character wise tokenization
File: `markov_chains.ipynb`  
Given last k-characters predicts immediate following character, and thus generates senntence.  
Generated samples:
```text

------------------------- GENERATED SENTENCE -- k = 1
oul mestothald schersheang y mens f
Any this
My INIsheff gemashelly waderearleisweinde towat s lleavi
------------------------- END

------------------------- GENERATED SENTENCE -- k = 2
EVAUREL:
Supoke the not night hiceiring offech beare eveign prale and, ing and mortessio.

Dide and th
------------------------- END

------------------------- GENERATED SENTENCE -- k = 3
kitch
The grow; but morship.

LUCIO:
I with alls,
To Baption ship clost that your quick moth to thou ho
------------------------- END

------------------------- GENERATED SENTENCE -- k = 4
? Prithmetimes not be sorrow the for than hope it for life,--for one's sometime bettery March times?

AR
------------------------- END

------------------------- GENERATED SENTENCE -- k = 5
ere England what's o'clock?

BRUTUS:
Do not so,
That walk in her nature,
The more; I'll desire that atten
------------------------- END

------------------------- GENERATED SENTENCE -- k = 8
at those my babies lulls asleep! the smiles on the adversaries,
Which, thou know'st it well they return.

AD
------------------------- END

------------------------- GENERATED SENTENCE -- k = 11
ng father:
The sister to my wife.

LUCENTIO:
And therefore hurt not: but
your people
But tie him not to be gone
------------------------- END

------------------------- GENERATED SENTENCE -- k = 14
, speak too:
Baccare! you are marvellous forward.

PETRUCHIO:
Why, therefore Warwick came to seek you out;
And the
------------------------- END
```


### Word-wise tokenization
File: `markov_chains.ipynb`  
Given last `k` words predicts the next immediate word, thus creating sentences.  
Sample generations:

```text
------------------------- GENERATED SENTENCE -- k = 1
preparation. 
 Reignier, Clarence did I will, 
 And welcome home again, in thy earliness doth give thee for my soul, 

 BAPTISTA: Come, make fit his troops, 
 If you of woe obey. 
 Thanks, wagery, lords, on Edward hath look' dst Polixenes; then I have goaded onward. 
 So season. 

 Is very Mab


 High- a state
------------------------- END

------------------------- GENERATED SENTENCE -- k = 2
s rattling bones, 
 The gods grant them true! 

 RICHARD: 
 These English woes will make them sharp, and, sooth to say I said loose- bodied and the time! 
 Henceforward do your best haste, his blood committed to your hand too much. Servants, leave me but love' s fight with none but thou talk' d Jove' s tooth doth never rankle more
 Than in my
------------------------- END

------------------------- GENERATED SENTENCE -- k = 3
near, be ne' er so fair, and I' ll pray a thousand prayers for thy death: 
' Tis he, that should be husband, comes to woo. 
 I' ll away before. 

 CAPULET: 
 Soft! take me with you, best brother?

 POLIXENES: 
 Then say at once what thou dost know in this. 
 Dost thou love hawking? thou hast hawks will
------------------------- END

------------------------- GENERATED SENTENCE -- k = 4
by guess. 

 KING HENRY VI: 
 I was moved withal. 

 DERBY: 
 Fortune and victory sit on thy helm! 

 RICHMOND: 
 God and your arms be praised, victorious friends, 
 The day is hot, the Capulets abroad, 
 And, touching hers, make blessed my rude hand. 
 Did my heart love till now? forswear it, sight! 
 For I ne' er
------------------------- END

------------------------- GENERATED SENTENCE -- k = 5
conceal' d lady to our cancell' d love?

 Nurse: 
 O, she is lame! love' s heralds should be thoughts, 
 Which ten times faster glide than the sun' s beams, 
 Driving back shadows over louring hills: 
 Therefore do nimble- pinion' d doves draw love, 
 And therefore hath the wind- swift Cupid wings. 
 Now is the sun upon the highmost hill
 Of
------------------------- END

------------------------- GENERATED SENTENCE -- k = 6
heart, poor sons, lamented for her, 
 Or the least votary I score my awe, 
 It breathes me of it. 

 CYMBELINE: 
 But those that I forsake old Gaunt, the earth to ever. 
 If they seem him and to yourself royal necessary
 Deliver me to my ancestors, and all regions
 It skill but Brutus was deep with the case. 

 Boy: 
 Why do you wring your hands
------------------------- END
```
We can see that by `k=6` it has completely overfitted.  



References:
- https://www.kdnuggets.com/2019/11/markov-chains-train-text-generation.html


----

# RNN

References:
- https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
- https://www.kaggle.com/code/abdulkhadarn/text-generation-with-character-based-rnn


----

